import argparse
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
import pickle
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from math import exp
from random import random
from skimage import io
import sys
from tqdm import tqdm

sys.path.append("./stylegan2-ada-pytorch")

# LFW functions taken from David Sandberg's FaceNet implementation
def distance(embeddings1, embeddings2, distance_metric=0):
    if distance_metric==0:
        # Euclidian distance
        diff = np.subtract(embeddings1, embeddings2)
        dist = np.sum(np.square(diff),1)
    elif distance_metric==1:
        # Distance based on cosine similarity
        dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1)
        norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
        similarity = dot / norm
        dist = np.arccos(similarity) / math.pi
    else:
        raise 'Undefined distance metric %d' % distance_metric

    return dist

class FaceReconsturction:
    
    def __init__(self, device='cuda'):
        self.device = device
        
        self.pregen_latents = None
        self.pregen_embeddings = None
        
        #
        # Load Networks
        #
        
        # Face recognition
        self.resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

        # Face detect/aling/crop
        self.mtcnn = MTCNN(
            image_size=160,
            margin=14,
            device=device,
            post_process=True,
        )

        # StyleGAN
        with open('ffhq.pkl', 'rb') as f:
            G = pickle.load(f)['G_ema'].to(device)  # torch.nn.Module
        self.G = G.eval()
        
    def gen_stylegan_latents(self, batch_size=8):
        z = torch.randn([batch_size, self.G.z_dim]).to(self.device)    # latent codes
        return z

    def infer_stylegan_faces(self, z):
        """
            z: latent vector(s) of dimension (batch_size, G.z_dim)
        """
        z = z.to(self.device)

        if len(z.shape) == 1:
            z = z.unsqueeze(0)

        with torch.no_grad():
            c = None                                # class labels (not used in this example)
            w = self.G.mapping(z, c, truncation_psi=0.5, truncation_cutoff=8)
            img = self.G.synthesis(w, noise_mode='const', force_fp32=True)
        return img


    def postprocess_stylegan_faces(self, img):
        """
        Applies following steps:
            - Changes dimension from (bs, 3, 1024, 1024) to (bs, 1024, 1024, 3)
            - Changes from [-1, 1] float to [0, 255] uint8 type
        """
        img_proc = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img_proc


    def get_facenet_embedding(self, img):
        """
            img: pytorch tensor of images of shape (batch_size, 1024, 1024, 3). 
                 Should be a tensor of type uint8, with integer values ranging from 0 to 255
        """

        assert img.shape[-1] == 3 and (len(img.shape) == 4 or len(img.shape) == 3)

        if type(img) == np.ndarray:
            img = torch.Tensor(img)

        if len(img.shape) == 3:
            img = img.unsqueeze(0)

        with torch.no_grad():
            im_crop = self.mtcnn(img.cpu())
            im_all = torch.cat([im.unsqueeze(0) for im in im_crop if im is not None], dim=0)
            img_embedding = self.resnet(im_all.to(self.device))

        return img_embedding


    def generate_and_save(self, n=8000, batch_size=16, file_name='pregenerated.pt'):
        iters = n // batch_size

        with torch.no_grad():
            latents_list = []
            embedding_list = []
            for i in tqdm(range(iters)):
                latents = self.gen_stylegan_latents(batch_size=batch_size)
                faces = self.infer_stylegan_faces(latents)
                faces_proc = self.postprocess_stylegan_faces(faces)
                embeddings = self.get_facenet_embedding(faces_proc)

                latents_list.append(latents.detach().cpu())
                embedding_list.append(embeddings.detach().cpu())

            all_latents = torch.cat(latents_list)
            all_embeddings = torch.cat(embedding_list)

        torch.save([all_latents, all_embeddings], file_name)

    def run_pregeneration_routine(self):
        """ 
        Pregenerates 160k (latent vector, embedding) pairs for faster initial matching.
        """
        for i in range(20):
            self.generate_and_save(n=8000, batch_size=16, file_name='pregenerated_'+str(i)+'.pt')

    def load_pregenerated(self):
        """
        Load pregenerated vectors
        
        Returns:
            all_latents: (n, 512) list of latent vectors
            all_embeddings: (n, 512) list of FaceNet embedding vectors
        """
        all_latents = []
        all_embeddings = []
        for i in range(20):
            latents, embeddings = torch.load('pregenerated/pregenerated_'+str(i)+'.pt')
            all_latents.append(latents)
            all_embeddings.append(embeddings)

        # Store result
        self.pregen_latents = torch.cat(all_latents)
        self.pregen_embeddings = torch.cat(all_embeddings)

    def find_closest_pregen(self, emb, pregen_latents, pregen_embeddings, offset=0):
        """
        Finds closest pregenerated match for given facenet embedding. Looks through
        a list of pregenerated (latent vector, embedding) pairs, and returns the 
        latent vector whose facenet embedding is the closest to the target.
        
        Args:
            emb: Target embedding to find closest latent vector to.
            pregen_latents: List of latent vectors
            pregen_embeddings: Associated list of embedding vectors
            
        Returns:
            (latent, embedding): tuple of latent vector and facenet embedding
        """
        norms = (pregen_embeddings-emb.to(pregen_embeddings)).norm(dim=1)
        sorted_idxs = norms.argsort()
        best_idx = sorted_idxs[offset]
        print('best norm: ', norms[best_idx])
        return pregen_latents[best_idx], pregen_embeddings[best_idx]


    def perform_face_reconstruction(self, target_emb, pregen=True, pregen_offset=0, init_zeros=False, 
                                    iters=400, use_annealing=False, std_multiplier=0.98):
        """Performs the face reconstruction algorithm

        Arguments:
            target_emb: torch.Tensor
                A Tensor of shape (1, 512), the target tensor to reconsturc a face for. 
            pregen: Boolean
                Whether to start from the closest pregenerated point.
                If false, will start from a random point.
            pregen_offset: int
                If pregen, will use n-th-to-best pregenerated match (default 0)
            init_zeros: Boolean
                If pregen is off, will either start at a zero vector (True), or a
                random nosie vector (False)
            iters: Number
                Number of iterations to run. More iteration generally mean better
                results, although tuning of std_multiplier is required
            use_annealing: Boolean
                Whether or not to use simulated annealing. If False, will instead
                use a greedy search.
            std_multiplier: float
                What factor to multiply the random noise by every iteration. The higher,
                the faster the algorithm will converge or stop improving.

        Returns:
            best_latent: torch.Tensor of shape (1, 512)
            best_list: list of images at each improvement
            best_norm_list: list of norms at each improvement
            best_latent_list: list of latent vectors at each improvement
            best_emb_list: list of embedding vectors at each improvement
        """
        
        def safe_exp(x):
            try: return exp(x)
            except: return 0

        def P(e, e_prime, T):

            if e_prime < e:
                return 1

            else:
                return safe_exp(-(e_prime-e)/T)

        # Make sure we're on gpu
        target_emb = target_emb.to(self.device)

        # Start from random point, or pregenerated match
        if init_zeros:
            best_latent = torch.zeros([1, self.G.z_dim]).cuda()
        else:
            best_latent = torch.randn([1, self.G.z_dim]).cuda() 
        best_norm = 1e7

        if pregen:
            if self.pregen_latents is None or self.pregen_embeddings is None:
                print("ERROR: Pregenerated latents and embeddings are not loaded")
                return
            
            best_latent, best_embedding = self.find_closest_pregen(target_emb, pregen_latents, pregen_embeddings, offset=pregen_offset)
            best_latent = best_latent.to(self.device)
            best_embedding = best_embedding.to(self.device)
            best_norm = (target_emb-best_embedding).norm()

        current_latent = best_latent
        current_norm = best_norm

        # Keep track of best images found
        best_list = []
        best_norm_list = []
        best_latent_list = []
        best_emb_list = []

        # Add the image of the starting latent to the list
        with torch.no_grad():
            best_face = self.postprocess_stylegan_faces(self.infer_stylegan_faces(current_latent))[0].cpu()
        best_list.append(best_face)
        best_norm_list.append(best_norm)
        best_latent_list.append(best_latent)
        best_emb_list.append(self.get_facenet_embedding(best_face))

        print('Starting norm: ', best_norm)
        std = 1
        T = 0

        with torch.no_grad():
            for i in range(iters):

                if use_annealing:
                    T = 1 - (i+1)/iters

                neighbor_latents = current_latent + std*torch.randn([16, self.G.z_dim]).cuda()
                faces = self.infer_stylegan_faces(neighbor_latents)
                faces_pp = self.postprocess_stylegan_faces(faces)
                embeddings = self.get_facenet_embedding(faces_pp)

                norms = (embeddings - target_emb).norm(dim=1)
                best_idx = torch.argmin(norms)

                if P(current_norm, norms[best_idx], T) > random():
                    if use_annealing:
                            print('[annealing] new current:', norms[best_idx].item())
                    current_latent = neighbor_latents[best_idx]
                    current_norm = norms[best_idx]

                if norms[best_idx] < best_norm:
                    print('New best:', norms[best_idx].item())
                    best_latent = neighbor_latents[best_idx]
                    best_norm = norms[best_idx]

                    best_list.append(faces_pp[best_idx].cpu())
                    best_norm_list.append(best_norm)
                    best_latent_list.append(best_latent)
                    best_emb_list.append(embeddings[best_idx])

            std *= std_multiplier

        return best_latent, best_list, best_norm_list, best_latent_list, best_emb_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--setup", action="store_true", help="Pregenerate 160k embeddings (stored in the same folder)")
    parser.add_argument("--pregen", action="store_true", help="Use pregenerated embedding to select a starting location")
    parser.add_argument("--anneal", action="store_true", help="Use simulated annealing in the reconstruction algorithm (by default, will use greedy algorithm.")
    parser.add_argument("--iters", type=int, default=400, help="Number of iterations")
    parser.add_argument("--img", type=str, help="Image path for reconstruction")
    parser.add_argument("--save_details", action="store_true", help="Save .pt file with detailed results")

    args = parser.parse_args()

    if args.pregen:
        print("Doing pregen")
    if args.anneal:
        print("Running with annealing")
    print(args)
        
    face_reconstruction = FaceReconsturction()
    
    if args.setup:
        print("Running setup procedure...")
        face_reconstruction.run_pregeneration_routine()
        
    else: 
        if args.pregen:
            face_reconstruction.load_pregenerated()

        image_paths = [args.img]
        images_pil = [Image.open(image_path).convert('RGB') for image_path in image_paths]
        images_pil_crop = [face_reconstruction.mtcnn(im).to('cuda') for im in images_pil]
        images_pil_crop_pp = [im.detach().cpu().permute(1, 2, 0)*0.5+0.5 for im in images_pil_crop]

        with torch.no_grad():
            target_embeddings = [face_reconstruction.resnet(im.unsqueeze(0)).cpu() for im in images_pil_crop]

        target_emb = target_embeddings[0]
        result = face_reconstruction.perform_face_reconstruction(target_emb, pregen=args.pregen, init_zeros=False, 
                                             use_annealing=args.anneal, iters=args.iters, std_multiplier=0.992)
        
        best_latent, best_list, best_norm_list, best_latent_list, best_emb_list = result
        im = Image.fromarray(best_list[-1].detach().cpu().numpy())
        im.save("output.png")
        
        if args.save_details:
            saved_output = {'targets': target_embeddings, 'results': results}
            torch.save(saved_output, 'result.pt')

