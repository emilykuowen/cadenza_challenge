import torch
from cadenza_system import CadenzaSystem

# Specify the path to the checkpoint file
checkpoint_path = 'cadenza_model_checkpoint_epoch0.pth'

# Load the checkpoint
checkpoint = torch.load(checkpoint_path)

# Access the model parameters
model_state_dict = checkpoint['model_state_dict']

# Access additional information if available
# For example, epoch, loss, etc.
epoch = checkpoint['epoch']
training_loss = checkpoint['training_losses']
validation_loss = checkpoint['validation_losses']

print("epoch", epoch)
print("training losses", training_loss)
print("validation losses", validation_loss)

# Now, you can use the loaded information as needed
# For example, to load the model and optimizer state:
# Create your model instance
model = CadenzaSystem()

# Load the model state
model.load_state_dict(model_state_dict)