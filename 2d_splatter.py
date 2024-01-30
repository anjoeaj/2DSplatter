
"""
Create a 2d Gaussian splatting reconstruction of an image.

Plan
1. Load image
2. Create a grid of points to splat onto
3. Create a Gaussian 2D renderer - define gaussian calculation logic here
4. Create a Gaussian 2D Model - define trainable parameters here. Define mean, covariance, rgb, alpha, theta, scale 
        TODO - Covariance calculation from 2d rot matrix
        Create a rot matrix. Multiply it with scale matrix and the transposes as shown inthe paper.
5. Optimizer - Adam
6. Loss - MSE, ssim
7. Train loop
8. Tensorboard logging
9. Export model and image

Challenges:
Covariance training - how to train covariance matrix.
Rerun covariance 
"""

import torch
import torchvision
import numpy as np
from torch.optim.lr_scheduler import MultiStepLR
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter
import imageio
from datetime import datetime

torch.autograd.set_detect_anomaly(False)
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

DEBUG = True

def circular_activation(input_tensor):
    # theta might overflow and cause NaNs. 
    # Here is an activation function to limit theta from 0 to 2pi
    # this function activates between 0 and 2 pi
    # +1 moves the sin range from [-1 to 1] to [0 to 2]
    
    # TODO - debug this more
    # return (1 + torch.sin(input)) * torch.pi
    return torch.sigmoid(input_tensor) * 2 * torch.pi

def load_image(filename):
    image = imageio.imread(filename)
    image = torch.tensor(image, dtype=torch.float32)
    image = image / 255
    return image

def write_image(filename, image):
    image = (image * 255).astype(np.uint8)
    print(f"Out of bounds warning -> min - {image.min()}, max - {image.max()}") if image.min() < 0 or image.max() > 255 else None
    create_path_if_not_exists(filename)
    imageio.imwrite(filename, image)

def create_path_if_not_exists(path):
    import os
    folder = os.path.dirname(path)
    if not os.path.exists(folder):
        os.makedirs(folder)
        
def export_video(image_folder,video_file):
    import subprocess

    frame_rate = 12
    command = f'ffmpeg -framerate {frame_rate} -i {image_folder}/%d.jpg -c:v libx264 -pix_fmt yuv420p {video_file}'
    subprocess.run(command, shell=True, check=True)

def check_for_nans(tensor, name):
    if torch.isnan(tensor).any() or torch.isinf(tensor).any():
        print(f"tensor {name} contains NaN or infinite values")
    
def save_model(model, filename):
    torch.save(model, filename)

def load_model(filename):
    model = torch.load(filename)
    return model

def debug_gaussians(gaussian_tensor,name, max_render=25):
    # gaussian_tensor = gaussian_tensor.detach().cpu().numpy()
    gaussian_tensor = gaussian_tensor.detach()
    # indices = torch.randperm(gaussian_tensor.shape[0])
    # indices = indices[:max_render] # pick random gaussians

    #override indices with 0-24 indices
    indices = torch.arange(0,max_render)
    grid = torchvision.utils.make_grid(gaussian_tensor[indices].permute(0,3,1,2), nrow=5)

    write_image(f"logimages/gaussian_tensor-{name}-{global_step}.jpg", grid.permute(1,2,0).cpu().numpy())

def debug_reconstructions(reconstructed_images, name, max_render=25):
    num_rows = int(torch.sqrt(torch.tensor(max_render)))
    grid = torchvision.utils.make_grid(reconstructed_images.permute(0,3,1,2), nrow=num_rows)
    write_image(f"logimages/reconstructed-{name}-{global_step}.jpg", grid.permute(1,2,0).cpu().numpy())

class GaussianRenderer2D():
    """
    This class renders a 2D image from a mixture of Gaussians.

    The render function renders the gaussian distributions onto a 2D plane.
    The class defines a grid of pixel coordinates to calculate the gaussian pdf.
    The pixels are initialized to cover a range from -1.0 to 1.0 in both x and y axes.
    It is then scaled by a factor of 3. 

    Attributes:
        render_plane_size (int): The size of the render plane (square).
        pixels (torch.Tensor): The pixel coordinates of the render plane.
        eps (float): A small value to avoid divide by zero errors.
    """
    def __init__(self, render_plane_size):
        self.render_plane_size = render_plane_size
        # self.height = render_plane_size[0]
        # self.width = render_plane_size[1]

        x = torch.linspace(-1.0, 1.0, steps=self.render_plane_size)
        y = torch.linspace(-1.0, 1.0, steps=self.render_plane_size)

	    # self.x, self.y = torch.meshgrid(x, y, indexing='ij') # check np.meshgrid indexing
        self.x, self.y = torch.meshgrid(x, y, indexing='xy')
        self.pixels = torch.stack([self.x, self.y], dim=-1).to(device) * 3
        self.eps = 1/255

    def render_plane(self):
        return self.pixels
    
    def gaussian_2d_pdf(self, pixels, mean, covariance):
        # calculate the gaussian pdf function
        # implement 2d multivariate gaussian pdf function
        # 𝐺(𝑥) = 𝑒 − 1 2 (𝑥 ) 𝑇 Σ −1 (𝑥 ) (from the paper)

        print(f"mean min - {mean.min()}, max - {mean.max()}") if DEBUG else None
        print(f"pixels min - {pixels.min()}, max - {pixels.max()}") if DEBUG else None

        # fix the shape by pytorch broadcasting
        pixel_diff = pixels[None,:,:,:,None] - mean[:,None,None,...]
        pixel_diff_r = pixel_diff.view(pixel_diff.shape[0], int((pixel_diff.shape[1]*pixel_diff.shape[2])) ,2,1)

        # TEST - run this below test to check if pdf is correct
        # pdf = torch.distributions.MultivariateNormal(mean[:,:,0], covariance)
        # x = pdf.log_prob(pixels.view(int((pixel_diff.shape[1]*pixel_diff.shape[2])),1,2))

        # TODO - check if this is correct
        prob = torch.exp(-0.5 * torch.transpose(pixel_diff_r, -1, -2) @ torch.inverse(covariance)[:,None,...] @ pixel_diff_r ) 
        print(f"prob min - {prob.min()}, max - {prob.max()}") if DEBUG else None
        # prob = prob / (2 * torch.tensor(torch.pi, device=device) * torch.sqrt(torch.det(covariance)[...,None,None,None]))

        return prob
    

    def render(self, mean, covariance, rgb, alpha, num_gaussians):

        mean = torch.tanh(mean) * 3
        splat = self.gaussian_2d_pdf(self.pixels, mean, covariance)
        print(f"covariance min - {covariance.min()}, max - {covariance.max()}") if DEBUG else None
        print(f"mean min - {mean.min()}, max - {mean.max()}") if DEBUG else None
        print(f"alpha min - {alpha.min()}, max - {alpha.max()}") if DEBUG else None
        check_for_nans(splat, "splat") if DEBUG else None

        splat = splat.reshape(-1, self.render_plane_size, self.render_plane_size, 1)
        print(f"splat min - {splat.min()}, max - {splat.max()}") if DEBUG else None

        #normaliztion tests on splat
        # splat = (splat - splat_min) / (splat_max - splat_min)
        splat = splat / torch.max(splat)
        print(f"splat norm min - {splat.min()}, max - {splat.max()}") if DEBUG else None

        debug_gaussians(splat,"gauss") if DEBUG else None
        check_for_nans(rgb, "rgb") if DEBUG else None

        rgb = torch.tanh(rgb) * 2
        rgb = (rgb + 1 ) / 2 # normalize to 0 to 1
        splat = splat * rgb[:,None,None,...]

        alpha_mask = torch.ones_like(splat) * torch.sigmoid(alpha[:,None,None,...]) # + self.eps
        check_for_nans(alpha_mask, "alpha_mask") if DEBUG else None
        
        print(f"alpha_mask min - {alpha_mask.min()}, max - {alpha_mask.max()}") if DEBUG else None
        masked_images = splat * alpha_mask
        debug_gaussians(masked_images, "masked") if DEBUG else None

        # Weighted sum of all gaussians based on alpha
        sum_masked_images = torch.sum(masked_images, dim=0) 

        blended_splat = torch.sigmoid(sum_masked_images)

        # TODO - try other blending approches blend based on alpha mask
        # something similar to the below approach
        # blended_splat = splat * alpha_mask + self.pixels * (1 - alpha_mask)


        # TODO take care of numerical stability of alpha (excerpt from the paper)
        # page 14 -> To address this, both in
        # the forward and backward pass, we skip any blending updates with
        # 𝛼 < 𝜖 (we choose 𝜖 as 1
        # 255 ) and also clamp 𝛼 with 0.99 from above.
        # Finally, before a Gaussian is included in the forward rasterization
        # pass, we compute the accumulated opacity if we were to include it
        # and stop front-to-back blending before it can exceed 0.9999.
        return blended_splat
    
class GaussianImage(torch.nn.Module):
    """
    This class represents an image as a mixture of Gaussians, each defined by a mean, rgb, alpha, theta, scale 

    There are five trainable parameters for each gaussian - mean, rgb, alpha, theta, scale
    They are initialized randomly and trained to reconstruct the target image.
    There is an option to initialize rgb pixels with the target image pixels based on the sampled mean.
    The forward function is the forward pass of the gaussian image optimization process. 
    It builds a covariance matrix from the trainable parameters and renders the gaussian image.

    Attributes:
        num_gaussians (int): The number of Gaussians to use in the mixture model.
        mean (torch.Tensor): The means of the Gaussians.
        alpha (torch.Tensor): The alpha values of the Gaussians.
        scale (torch.Tensor): The scales of the Gaussians.
        theta (torch.Tensor): The rotation angles of the Gaussians.
        rgb (torch.Tensor): The RGB color values of the Gaussians.
        renderer (GaussianRenderer2D): The renderer used to render the image.
    """
    def __init__(self, num_gaussians, render_plane_size, init_from_rgb=False):
        super().__init__()
        self.num_gaussians = num_gaussians

        # define trainable parameters
        self.mean = torch.rand((self.num_gaussians, 2, 1)).to(device)
        self.mean = self.mean * 2 - 1
        self.mean = torch.nn.Parameter(self.mean, requires_grad=True) 

        self.alpha = torch.nn.Parameter(torch.rand((self.num_gaussians, 1)).to(device), requires_grad=True)
        self.scale = torch.nn.Parameter((torch.rand(self.num_gaussians, 2)).to(device), requires_grad=True)
        # angle of rotation on the plane
        self.theta = torch.nn.Parameter(torch.rand((self.num_gaussians, 1)).to(device), requires_grad=True)

        self.renderer = GaussianRenderer2D(render_plane_size)

        # initialize rgb with the target image samples
        if init_from_rgb:
            grid = self.mean.unsqueeze(0).view(1, self.num_gaussians, 1, 2)
            # sample the target image at the gaussian mean points. 
            sampled_image = torch.nn.functional.grid_sample(target_image.unsqueeze(0).permute(0,3,1,2), grid) 
            sampled_image = sampled_image.squeeze(0).squeeze(2)

            self.rgb = torch.nn.Parameter(sampled_image.permute(1,0).to(device), requires_grad=True)
        else:
            self.rgb = torch.nn.Parameter(torch.rand((self.num_gaussians, 3)).to(device), requires_grad=True)

        self.myparamlist = torch.nn.ParameterList([self.mean, self.rgb, self.alpha, self.scale, self.theta])

    def build_covariance_matrix(self):
        # TODO - Covariance calculation from 2d rot matrix
        # Create a 2d rot matrix. Scale it with a scale matrix
        # [cos(theta), -sin(theta)] 
        # [sin(theta), cos(theta)]
        # Σ = 𝑅𝑆𝑆_𝑇 𝑅_T

        # theta might overflow and cause NaNs.
        theta_activated = circular_activation(self.theta)

        cos_theta = torch.cos(theta_activated)
        sin_theta = torch.sin(theta_activated)

        # build a 2x2 tensor with cos(theta) and sin(theta)
        rot_mat = torch.stack([
            torch.concat([cos_theta, -sin_theta], dim=-1),
            torch.concat([sin_theta, cos_theta], dim=-1)
        ], dim=-2)

        print(f"rot_mat min - {rot_mat.min()}, max - {rot_mat.max()}") if DEBUG else None
        print(f"scale min - {self.scale.min()}, max - {self.scale.max()}") if DEBUG else None
        
        # TODO - Debug sigmoid activations for scale
        # scale_activated = torch.sigmoid(self.scale)
        scale_activated = self.scale
        # scale_activated = torch.clamp(self.scale,0.01,4.0)

        # create diagonal scale matrix
        scale_mat = torch.eye(scale_activated.shape[1]).to(device).unsqueeze(0) * scale_activated.unsqueeze(-1)

        scale_transpose = torch.transpose(scale_mat,-1,-2)
        rot_transpose = torch.transpose(rot_mat,-1,-2)
        covariance = rot_mat @ scale_mat @ scale_transpose @ rot_transpose

        # add a small jitter to the covariance matrix to make it invertible
        # Not having this causes NaNs in the gaussian pdf calculation randomly
        jitter = 1e-6 * torch.eye(covariance.shape[-1]).to(device).unsqueeze(0)
        covariance = covariance + jitter

        return covariance
    
    def forward(self):
        
        covariance = self.build_covariance_matrix()
        gaussian_reconstruction = self.renderer.render(self.mean, covariance, self.rgb, self.alpha, self.num_gaussians)

        return gaussian_reconstruction
    
num_gaussians = 1500
render_plane_size = 128
num_steps = 3000

curr_time = datetime.now().strftime("%Y%m%d-%H%M%S")
writer = SummaryWriter(f"runs/{curr_time}") # tensorboard logging init

target_image = load_image("trinity.jpg").to(device)

# resize the image to render plane size
target_image = torch.nn.functional.interpolate(target_image.unsqueeze(0).permute(0,3,1,2), size=render_plane_size)
target_image = target_image.squeeze(0).permute(1,2,0)

gaussian_image = GaussianImage(num_gaussians, render_plane_size).to(device)
optimizer = torch.optim.Adam(gaussian_image.parameters(), lr=0.01)
scheduler = MultiStepLR(optimizer, milestones=[400,1200], gamma=0.8)

l1_loss = torch.nn.L1Loss().to(device)

writer.add_image("Target Image", target_image.detach().permute(2,0,1), 0) # permute to make it CHW

global_step = 0
reconstructions = torch.empty(0, *target_image.shape).to(device)
for step in range(num_steps):
    global_step = step
    optimizer.zero_grad()
    reconstructed_image = gaussian_image()
    loss = l1_loss(reconstructed_image, target_image) 

    # TODO - Try SSIM loss
    # loss = loss + pytorch_ssim.ssim(reconstructed_image.permute(2,0,1).unsqueeze(0), target_image.permute(2,0,1).unsqueeze(0)).to(device)

    loss.backward()

    # TODO Try clipping gradients
    # torch.nn.utils.clip_grad_norm_(gaussian_image.scale, 1)
    # torch.nn.utils.clip_grad_value_(gaussian_image.alpha, 0.00392) # 1/255

    optimizer.step()
    scheduler.step()
    reconstructions = torch.cat((reconstructions, reconstructed_image.detach().unsqueeze(0)), 0)

    if DEBUG and (step + 1) % 25 == 0:
        debug_reconstructions(reconstructions,"", 25)
        reconstructions = torch.empty(0, *target_image.shape).to(device)

    print(f">>>> Step {step} loss: {loss} <<<<")
    if step % 30 == 0:
        writer.add_scalar("loss", loss, step)
        detached_img = reconstructed_image.detach()
        writer.add_image("Gaussian Image", detached_img.permute(2,0,1), step) # permute to make it CHW
        write_image(f"runs_videos/{curr_time}/frames/{step}.jpg", detached_img.cpu().numpy())


# Save the model
save_model(gaussian_image, "gaussian_image.pt")

# Save the reconstructed image to disk
write_image(f"runs_videos/{curr_time}/reconstructed_image.png", reconstructed_image.detach().cpu().numpy())
write_image(f"runs_videos/{curr_time}/target_image.png", target_image.detach().cpu().numpy())


