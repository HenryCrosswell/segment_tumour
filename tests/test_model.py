import torch
from model import TumorUNet

def test_tumour_unet_output_shape():
    # Create a UNet model with 1 input channel and 4 output channels (for segmentation)
    model = TumorUNet(in_channels=1, out_channels=4)
    
    # Create a random input tensor with the specified shape (batch_size=1, channels=1, height=128, width=128)
    input_tensor = torch.rand(1, 1, 128, 128)
    
    # Forward pass through the model
    output = model(input_tensor)
    
    # Check the output shape
    assert output.shape == torch.Size([1, 4, 128, 128])

def test_tumour_unet_encoding():
    # Create a UNet model with 1 input channel and 4 output channels (for segmentation)
    model = TumorUNet(in_channels=1, out_channels=4)
    
    # Create a random input tensor with the specified shape (batch_size=1, channels=1, height=128, width=128)
    input_tensor = torch.rand(1, 1, 128, 128)
    
    # Forward pass through the model to get the intermediate encoding
    encoding = model.encoder(input_tensor)
    
    # Check the encoding shape after the encoder part
    assert encoding.shape == torch.Size([1, 64, 64, 64])

def test_tumour_unet_decoding():
    # Create a UNet model with 1 input channel and 4 output channels (for segmentation)
    model = TumorUNet(in_channels=1, out_channels=4)
    
    # Create a random input tensor with the specified shape (batch_size=1, channels=1, height=128, width=128)
    input_tensor = torch.rand(1, 1, 128, 128)
    
    # Forward pass through the model to get the output after the decoder part
    output = model(input_tensor)
    
    # Check the output shape after the decoder part
    assert output.shape == torch.Size([1, 4, 128, 128])

def test_tumour_unet_forward():
    # Create a UNet model with 1 input channel and 4 output channels (for segmentation)
    model = TumorUNet(in_channels=1, out_channels=4)
    
    # Create a random input tensor with the specified shape (batch_size=1, channels=1, height=128, width=128)
    input_tensor = torch.rand(1, 1, 128, 128)
    
    # Forward pass through the model
    output = model(input_tensor)
    
    # Check the output shape after the forward pass
    assert output.shape == torch.Size([1, 4, 128, 128])
