import torch
from cadenza_system import CadenzaSystem
import matplotlib.pyplot as plt
import numpy as np

training_loss_array = []
validation_loss_array = []

for i in range(15):
    # Specify the path to the checkpoint file
    checkpoint_path = 'checkpoints/cadenza_model_checkpoint_epoch' + str(i) + '.pth'

    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path)

    # Access additional information if available
    # For example, epoch, loss, etc.
    epoch = checkpoint['epoch']
    training_loss_array += checkpoint['training_losses']
    validation_loss_array += checkpoint['validation_losses']

    print("epoch", epoch)
    # print("training losses", training_loss)
    # print("validation losses", validation_loss)

print(len(training_loss_array))

plt.plot(training_loss_array)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.savefig('training_loss.png')
plt.show()

plt.plot(validation_loss_array)
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.savefig('validation_loss.png')
plt.show()

# Now, you can use the loaded information as needed
# For example, to load the model and optimizer state:
# Create your model instance
# model = CadenzaSystem()

# Access the model parameters
# model_state_dict = checkpoint['model_state_dict']

# # Load the model state
# model.load_state_dict(model_state_dict)