
"""
Create a 2d Gaussian splatting recosntruction of an image.

Plan
1. Load image
2. Create a grid of points to splat onto
3. Create a Gaussian 2D renderer - define gaussian calculation logic here
4. Create a Gaussian 2D Model - define trainable parameters here. Define mean, covariance, covariance, rgba. 
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
import pytorch_ssim
from torch.utils.tensorboard import SummaryWriter
import imageio
from datetime import datetime

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

def circular_activation(input):
    # theta might overflow and cause NaNs. 
    # Here is an activation function to limit theta from 0 to 2pi
    # this function activates between 0 and 2 pi
    # +1 moves the sin range from [-1 to 1] to [0 to 2]

    return (1 + torch.sin(input)) * torch.pi
    # return torch.sigmoid(input) * 2 * torch.pi

def save_model(model, filename):
    torch.save(model, filename)

def load_model(filename):
    model = torch.load(filename)
    return model

class GaussianRenderer2D():
    def __init__(self, render_plane_size):
        self.render_plane_size = render_plane_size
        # self.height = render_plane_size[0]
        # self.width = render_plane_size[1]

        x = torch.linspace(0, 1, steps=self.render_plane_size)
        y = torch.linspace(0, 1, steps=self.render_plane_size)

	    # self.x, self.y = torch.meshgrid(x, y, indexing='ij') # check np.meshgrid indexing
        self.x, self.y = torch.meshgrid(x, y, indexing='xy')
        self.pixels = torch.stack([self.x, self.y], dim=-1).to(device)
        self.eps = 1/255

    def render_plane(self):
        return self.pixels
    
    def gaussian_2d_pdf(self, pixels, mean, covariance):
        # calculate the gaussian pdf function
        # implement 2d multivariate gaussian pdf function
        # https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Bivariate_case
        # 𝐺(𝑥) = 𝑒 − 1 2 (𝑥 ) 𝑇 Σ −1 (𝑥 )

        # prob = torch.exp(-0.5 * torch.transpose(pixel - mean) @ torch.inverse(covariance) @ (pixel - mean)) / (2 * torch.tensor(torch.pi, device=device) * torch.sqrt(torch.det(covariance)))
        
        #TODO - unwrap pixels

        # fix the shape by pytorch broadcasting
        #pixel_diff.shape
        # torch.Size([10, 64, 64, 2])
        pixel_diff = pixels[None, ...] - mean[:, None, None, :] 
        pixel_diff_r = pixel_diff.view(pixel_diff.shape[0], int((pixel_diff.shape[1]*pixel_diff.shape[2])) ,1,-1)


        # prob = torch.exp(-0.5 * torch.transpose(pixel_diff_r, -1, -2) @ torch.inverse(covariance)[:,None,...] @ pixel_diff_r) 

        #TODO - hack to fix the shape by swapping the order of multiplication.
        #TODO - check if this is correct
        prob = torch.exp(-0.5 * pixel_diff_r @ torch.inverse(covariance)[:,None,...] @  torch.transpose(pixel_diff_r, -1, -2)) 


        # prob /= 2 * torch.tensor(torch.pi, device=device) * torch.sqrt(torch.det(covariance))
        return prob
    

    def render(self, mean, covariance, rgb, alpha, num_gaussians):

        splat = self.gaussian_2d_pdf(self.pixels, mean, covariance)
        splat = splat.reshape(-1, self.render_plane_size, self.render_plane_size,1)
        splat = splat * rgb[-1,None,None,...]

        alpha_mask = torch.ones_like(splat) * alpha[-1,None,None,...]
        masked_images = splat * alpha_mask

        blended_splat = torch.sum(masked_images, dim=0)

        # blend based on alpha mask
        # blended_splat = splat * alpha_mask + self.pixels * (1 - alpha_mask)


        # take care of numerical stability of alpha 
        # page 14 -> To address this, both in
        # the forward and backward pass, we skip any blending updates with
        # 𝛼 < 𝜖 (we choose 𝜖 as 1
        # 255 ) and also clamp 𝛼 with 0.99 from above.
        # Finally, before a Gaussian is included in the forward rasterization
        # pass, we compute the accumulated opacity if we were to include it
        # and stop front-to-back blending before it can exceed 0.9999.
        return blended_splat
    
class GaussianImage(torch.nn.Module):
    def __init__(self, num_gaussians, render_plane_size):
        super().__init__()
        self.num_gaussians = num_gaussians

        # define trainable parameters
        self.mean = torch.nn.Parameter(torch.rand((self.num_gaussians, 2)), requires_grad=True).to(device) 
        # self.covariance = torch.nn.Parameter(torch.rand((self.num_gaussians, 2, 2)), requires_grad=True)
        self.rgb = torch.nn.Parameter(torch.rand((self.num_gaussians, 3)), requires_grad=True).to(device)
        self.alpha = torch.nn.Parameter(torch.rand((self.num_gaussians, 1)), requires_grad=True).to(device)
        self.scale = torch.nn.Parameter(torch.rand((self.num_gaussians, 2)), requires_grad=True).to(device)
        # angle of rotation on the plane
        self.theta = torch.nn.Parameter(torch.rand((self.num_gaussians, 1)), requires_grad=True).to(device)

        self.renderer = GaussianRenderer2D(render_plane_size)

        self.covariance = self.build_covariance_matrix()

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
        # TODO - check if this is the right way to build a 2x2 tensor without losing sin and cos
        rot_mat = torch.stack([
            torch.concat([cos_theta, -sin_theta], dim=-1),
            torch.concat([sin_theta, cos_theta], dim=-1)
        ], dim=-2)
        # rot_mat = torch.tensor([[torch.cos(theta_activated), -torch.sin(theta_activated)], [torch.sin(theta_activated), torch.cos(theta_activated)]])

        # create diagonal scale matrix
        scale_mat = torch.eye(self.scale.shape[1]).unsqueeze(0) * self.scale.unsqueeze(-1)

        scale_transpose = torch.transpose(scale_mat,-1,-2)
        rot_transpose = torch.transpose(rot_mat,-1,-2)
        covariance = rot_mat @ scale_mat @ scale_transpose @ rot_transpose

        return covariance
    
    def forward(self):
        
        gaussian_reconstruction = self.renderer.render(self.mean, self.covariance, self.rgb, self.alpha, self.num_gaussians)
        # gaussian_reconstruction = torch.rand((num_gaussians,self.renderer.render_plane_size, self.renderer.render_plane_size, 3))

        return gaussian_reconstruction
    
# new_img = GaussianImage(10)
num_gaussians = 10
render_plane_size = 64
num_steps = 100

# Sets up a timestamped log directory.
logdir = "logs/" + datetime.now().strftime("%Y%m%d-%H%M%S")
# Creates a file writer for the log directory.
writer = SummaryWriter(log_dir=logdir)

target_image = torch.rand((render_plane_size, render_plane_size, 3)).to(device)
gaussian_image = GaussianImage(num_gaussians, render_plane_size).to(device)
optimizer = torch.optim.Adam(gaussian_image.parameters(), lr=0.01)
l1_loss = torch.nn.L1Loss()
writer.add_image("Target Image", target_image.permute(2,0,1), 0) # permute to make it CHW

for step in range(num_steps):
    reconstructed_image = gaussian_image()
    loss = l1_loss(reconstructed_image, target_image) 
    # loss += pytorch_ssim.ssim(reconstructed_image.unsqueeze(0), target_image.unsqueeze(0))

    # loss.backward(retain_graph=True)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"Step {step} loss: {loss}")
    if step % 10 == 0:
        writer.add_scalar("loss", loss, step)
        # writer.add_image("Reconstructed Image", reconstructed_image.detach().permute(2,0,1), step)

# Save the model
save_model(gaussian_image, "gaussian_image.pt")

# Save the reconstructed image to disk
imageio.imwrite("reconstructed_image.png", reconstructed_image)

