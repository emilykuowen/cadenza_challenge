import torch
import logging
import auraloss
from itertools import chain

from deepafx_st.utils import system_summary
from cadenza_model import CadenzaModel
from cadenza_dataset import CadenzaDataset

if __name__ == "__main__":
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # create the System
    cadenza_model = CadenzaModel().to(device)

    # print details about the model
    system_summary(cadenza_model)

    root_directory = "/data/home/ubuntu/data/cad_icassp_2024"
    train_dataset = CadenzaDataset(root_directory, subset='train', duration=10)
    valid_dataset = CadenzaDataset(root_directory, subset='valid', duration=10)

    batch_size = 8
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        chain(
            cadenza_model.encoder.parameters(),
            cadenza_model.processor.parameters(),
            cadenza_model.controller.parameters(),
        ),
        lr=1e-4,
        betas=(0.9, 0.999),
    )

    recon_losses = torch.nn.ModuleDict()
    recon_losses["mrstft"] = auraloss.freq.MultiResolutionSTFTLoss(
            fft_sizes=[32, 128, 512, 2048, 8192, 32768],
            hop_sizes=[16, 64, 256, 1024, 4096, 16384],
            win_lengths=[32, 128, 512, 2048, 8192, 32768],
            w_sc=0.0,
            w_phs=0.0,
            w_lin_mag=1.0,
            w_log_mag=1.0,
        )
    recon_losses["l1"] = torch.nn.L1Loss()
    
    # Training loop
    num_epochs = 10  # Adjust the number of epochs as needed
    data_sample_rate = 44100

    training_losses, validation_losses = [], []

    for epoch in range(num_epochs):
        print("Epoch ", epoch)
        
        # Training phase
        cadenza_model.train()
        
        iteration = 0        
        for batch in train_loader:
            original_tensor, reference_tensor, gain_tensor = batch
            original_tensor, reference_tensor = original_tensor.unsqueeze(1), reference_tensor.unsqueeze(1)
            original_tensor.requires_grad_()
            reference_tensor.requires_grad_()
            original_tensor.to(device)
            reference_tensor.to(device)
            gain_tensor.to(device)
            
            # Forward pass
            output_tensor, p_with_gain, e_x = cadenza_model(x=original_tensor, y=reference_tensor, gain=gain_tensor, data_sample_rate=data_sample_rate)

            training_loss = 0
            recon_loss_weights = [1, 100]
            
            # compute reconstruction loss terms
            for loss_idx, (loss_name, recon_loss_fn) in enumerate(recon_losses.items()):
                recon_loss = recon_loss_fn(output_tensor, reference_tensor)  # reconstruction loss
                recon_loss_weight = float(recon_loss_weights[loss_idx])
                training_loss += recon_loss_weight * recon_loss

            training_loss /= batch_size
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{iteration+1}/{len(train_loader)}], Training Loss: {training_loss}')
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{iteration+1}/{len(train_loader)}], Training Loss: {training_loss}')
            iteration += 1

            # Backward pass and optimization
            training_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            training_losses.append(training_loss.item())

        # Validation phase
        cadenza_model.eval()

        iteration = 0
        with torch.no_grad():
            for batch in valid_loader:
                original_tensor, reference_tensor, gain_tensor = batch
                original_tensor.to(device)
                reference_tensor.to(device)
                gain_tensor.to(device)

                output_tensor, p_with_gain, e_x = cadenza_model(x=original_tensor, y=reference_tensor, gain=gain_tensor, data_sample_rate=data_sample_rate)
                
                recon_loss_weights = [1, 100]
                validation_loss = 0
                # compute reconstruction loss terms
                for loss_idx, (loss_name, recon_loss_fn) in enumerate(recon_losses.items()):
                    recon_loss = recon_loss_fn(output_tensor, reference_tensor)  # reconstruction loss
                    recon_loss_weight = float(recon_loss_weights[loss_idx])
                    validation_loss += recon_loss_weight * recon_loss

                validation_loss /= batch_size
                validation_losses.append(validation_loss)
                logger.info(f'Epoch [{epoch+1}/{num_epochs}], Batch [{iteration+1}/{len(valid_loader)}], Validation Loss: {validation_loss}')
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{iteration+1}/{len(valid_loader)}], Validation Loss: {validation_loss}')
                iteration += 1
            
        checkpoint_info = {
            'epoch': epoch,
            'training_losses': training_losses,
            'validation_losses': validation_losses
        }

        # Create a dictionary to save both the model state and additional information
        checkpoint = {
            'model_state_dict': cadenza_model.state_dict(),
            **checkpoint_info,
        }

        # Save the checkpoint to a file
        checkpoint_path = 'checkpoints/cadenza_dataset_epoch' + str(epoch) + '.pth'
        torch.save(checkpoint, checkpoint_path)

        logger.info(f"Checkpoint saved at {checkpoint_path}")
        print(f"Checkpoint saved at {checkpoint_path}")
