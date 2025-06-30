import torch
from torch.utils.data import Dataset
from torch import nn
import polars as pl
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


class BlockDataset(Dataset):
    def __init__(self, features, creator_idx, labels=None):
        """
        Dataset class that works with both training and test sets.

        Args:
            features: Feature tensors
            creator_idx: Creator encoding indices
            labels: Optional labels (None for test sets)
        """
        self.features = features
        self.creators = creator_idx
        self.labels = labels
        self.is_test = labels is None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.is_test:
             # Return feature and creator index (no labels during test)
            return self.features[idx], self.creators[idx]
        else:
            # Return feature, creator index, and label for training/validation
            return self.features[idx], self.creators[idx], self.labels[idx]


class LoadDataFrames:
    """
    Class to handle loading and managing various data files for the Solana prediction model.
    
    This class provides methods to load training data, static features, test data, and creator encodings
    from parquet files. It maintains references to all loaded dataframes for easy access.
    """
    def __init__(self, train_df_path='TRAIN_min_max_features.parquet', static_df_path='combined_static_features_train.parquet', test_df_path='TEST_min_max_features.parquet', creator_encoding_path='creator_encodings.parquet'):
        self.train_df_path = train_df_path
        self.static_df_path = static_df_path
        self.creator_encoding_path = creator_encoding_path
        self.test_df_path = test_df_path
        # Initialize dataframe placeholders
        self.train_df = None
        self.static_df = None
        self.test_df = None
        self.creator_encoding_df = None

    def load_data(self):
        self.train_df = pl.read_parquet(self.train_df_path)
        self.static_df = pl.read_parquet(self.static_df_path).select(pl.col(['mint', 'has_graduated'])).with_columns(pl.col('has_graduated').cast(pl.Int64))
        self.creator_encoding_df = pl.read_parquet(self.creator_encoding_path)

    def load_test_data(self):
        self.test_df = pl.read_parquet(self.test_df_path)
        return self.test_df

    def test_data_tokens(self):
        """
        Get unique tokens (base_coin values) from the test dataset
        """
        return self.test_df.select('base_coin').unique(keep="first", maintain_order=True)



def create_block_stack(features_df, static_df, creator_encodings,seq_len=100, is_test=False, stop=None):
    """
    Create feature stacks for training or test data by processing sequential blocks.
    
    processes the features dataframe in chunks of seq_len rows, creating
    feature tensors and corresponding creator indices. For training data, it also
    retrieves labels from the static features dataframe.

    Args:
        features_df: Polars DataFrame containing time-series features
        static_df: Polars DataFrame containing static features with labels
        creator_encodings: Creator encoding mappings (currently unused but kept for compatibility)
        seq_len: Sequence length for each block (default: 100)
        is_test: Whether this is test data without labels (default: False)
        stop: Optional stopping point for processing (default: None)

    Returns:
        For training data: (features_tensor, creator_indices_tensor, labels_tensor)
        For test data: (features_tensor, creator_indices_tensor)
    """


    # Get feature columns (excluding metadata columns)
    # columns contain the actual features used for model training/prediction
    feature_cols = [col for col in features_df.columns
                    if col not in ['base_coin', 'norm_slot', 'creator_final', 'mint', 'slot', 'creator_final_right', 'creator_encoding']]

    # Process all blocks
    token_blocks = []
    total_rows = features_df.height

    # Calculate total iterations for progress bar
    total_iterations = total_rows // seq_len
    if stop is not None:
        total_iterations = min(total_iterations, stop // seq_len + 1)

    # Create progress bar to track block creation progress
    pbar = tqdm(range(0, total_rows, seq_len), desc="Creating blocks", total=total_iterations)

    for i in pbar:
        # Skip incomplete blocks if needed (ensure we have full seq_len blocks)
        if i + seq_len > total_rows:
            continue

        # Get the block data (seq_len consecutive rows)
        block = features_df.slice(i, seq_len)

        # Get base_coin (should be the same for all rows in block)
        # identifies which token/coin this block belongs to
        base_coin = block['base_coin'][0]

        # Update progress bar
        if i % 10000 == 0:
            pbar.set_description(f"Processing block {i / 100}")

        # Extract creator encoding directly from the block
        # These indices map creators to their encoded representations
        creator_indices = torch.tensor(block['creator_encoding'].to_numpy(), dtype=torch.int)

        # Convert feature columns to tensor for model input
        features = torch.tensor((block.select(feature_cols).to_numpy())).type(torch.float)

        if is_test:
            # For test data store features and creator index (no labels available)
            token_blocks.append((features, creator_indices))
        else:
            # For training data, include label if available
            # Look up the graduation status for this token from static features
            label = static_df.filter(pl.col('mint') == base_coin).select('has_graduated').item()

            if label is not None:
                # Convert label to tensor format
                label = torch.tensor([label], dtype=torch.float)
                token_blocks.append((features, creator_indices, label))

        # Stop processing if we've reached the specified limit
        if stop is not None and i >= stop:
            break

    # Validate that we have valid blocks to work with
    if len(token_blocks) == 0:
        raise ValueError("No valid blocks found! Check your data and mapping dictionaries.")

    # Stack all features into a single tensor (batch_size, seq_len, num_features)
    features = torch.stack([block[0] for block in token_blocks])
    # Stack all creator indices into a single tensor (batch_size, seq_len)
    creator_indices = torch.stack([block[1] for block in token_blocks])

    if is_test:
        # For test data return features and creator indices only
        print(f"Created test dataset with {len(token_blocks)} blocks")
        return features, creator_indices
    else:
        # For training data, also stack labels (batch_size, 1)
        labels = torch.stack([block[2] for block in token_blocks])
        print(f"Created training dataset with {len(token_blocks)} blocks")
        return features, creator_indices, labels



def train(model, training_data, validation_data=None,n_epoch = 1, report_every = 10, learning_rate = 3e-4,
          criterion = nn.BCEWithLogitsLoss(), optimizer=torch.optim.AdamW, device='cpu',
          threshold=0.5, early_stopping_patience=None):
    """
    Train a PyTorch model with optional validation and early stopping.

    Args:
        model: PyTorch model to train
        training_data: DataLoader containing training batches
        validation_data: Optional DataLoader for validation (default: None)
        n_epoch: Number of training epochs (default: 1)
        report_every: Report metrics every N epochs (default: 10)
        learning_rate: Learning rate for optimizer (default: 3e-4)
        criterion: Loss function (default: BCEWithLogitsLoss)
        optimizer: Optimizer class (default: AdamW)
        device: Device to train on ('cpu' or 'cuda', default: 'cpu')
        threshold: Classification threshold for binary predictions (default: 0.5)
        early_stopping_patience: Stop early if validation doesn't improve for N epochs (default: None)

    Returns:
        Dictionary containing training history:
        - 'train_loss': List of training losses per epoch
        - 'val_loss': List of validation losses per epoch (if validation_data provided)
        - 'val_metrics': List of validation metrics per epoch (if validation_data provided)
    """
    # Initialize tracking variables
    all_train_losses = []
    all_val_losses = []
    all_val_metrics = []

    # Setup model and optimizer
    model.to(device)
    model.train()
    optimizer = optimizer(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"Training on dataset with {len(training_data)} batches")
    if validation_data:
        print(f"Using validation set with {len(validation_data)} batches")

    # Create epoch progress bar
    epoch_pbar = tqdm(range(1, n_epoch + 1), desc="Epochs", position=0)

    for epoch in epoch_pbar:
        # Reset gradients and tracking variables for new epoch
        model.zero_grad()  # clear the gradients
        epoch_loss = 0.0
        batch_count = 0

        # Create training batch progress bar
        train_pbar = tqdm(training_data, desc=f"Epoch {epoch}/{n_epoch} training",
                         leave=False, position=1)

        # Training loop for current epoch
        for i, (block, x_idx, label) in enumerate(train_pbar):
            # Move data to specified device
            block, x_idx, label = block.to(device), x_idx.to(device), label.to(device)
            model.train()

            # print(block.shape)

            block, x_idx, label = block.to(device), x_idx.to(device), label.to(device)

            # Forward pass through model
            logit_output = model(block, x_idx)

            # Calculate loss
            loss = criterion(logit_output, label)

            # Backward pass and optimization
            loss.backward()

            # clip gradients to prevent exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), 3)
            optimizer.step()
            optimizer.zero_grad()

            # Accumulate loss for this epoch
            epoch_loss += loss.item()
            batch_count += 1

            # Update training progress bar with current loss (avoid too frequent updates)
            if i % 1000000 == 0:  # Update every 10 batches to avoid slowdown
                train_pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Calculate average loss for the epoch
        avg_train_loss = epoch_loss / batch_count
        all_train_losses.append(avg_train_loss)

        # validation phase (if validation data is provided)
        if validation_data:
            model.eval()  # Set model to evaluation mode
            val_loss = 0.0
            val_batch_count = 0
            val_correct = 0
            val_total = 0
            val_tp, val_fp, val_fn = 0, 0, 0  # True positives, false positives, false negatives
            # Store raw probabilities and labels during validation for metric calculation
            all_probs = []
            all_labels = []

            # Create validation batch progress bar
            val_pbar = tqdm(validation_data, desc=f"Epoch {epoch}/{n_epoch} validation",
                           leave=False, position=1)

            with torch.no_grad():  # Disable gradient for  validation
                for i, (val_block, val_idx, val_label) in enumerate(validation_data):
                    # Move validation data to device
                    val_block, val_idx, val_label = val_block.to(device), val_idx.to(device), val_label.to(device)

                    # Forward pass through model
                    val_output = model(val_block, val_idx)
                    # Calculate validation loss
                    batch_loss = criterion(val_output, val_label)
                    val_loss += batch_loss.item()
                    val_batch_count += 1

                    # Calculate predictions and probabilities
                    val_probs = torch.sigmoid(val_output)
                    val_preds = val_output.argmax(dim=1)

                    # print('val_prediction',val_preds)

                    # Store raw probabilities and labels for comprehensive metric calculation
                    all_probs.append(val_probs.cpu().detach())
                    all_labels.append(val_label.cpu().detach())

                    # Calculate accuracy metrics
                    val_correct += (val_preds == val_label).float().sum().item()
                    val_total += val_label.numel()

                    # Calculate confusion matrix components for precision/recall
                    val_tp += torch.logical_and(val_preds == 1, val_label == 1).sum().item()
                    val_fp += torch.logical_and(val_preds == 1, val_label == 0).sum().item()
                    val_fn += torch.logical_and(val_preds == 0, val_label == 1).sum().item()

                    # Update validation progress bar occasionally
                    if i % 10 == 0:  # Update every 10 batches
                        val_pbar.set_postfix(loss=f"{batch_loss.item():.4f}")

            # Concatenate all probabilities and labels
            all_probs = torch.cat(all_probs, dim=0).numpy()
            all_labels = torch.cat(all_labels, dim=0).numpy()

            # Calculate comprehensive validation metrics
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            val_accuracy = val_correct / val_total if val_total > 0 else 0
            val_precision = val_tp / (val_tp + val_fp) if (val_tp + val_fp) > 0 else 0

            # Calculate AUC
            val_auc_roc = roc_auc_score(all_labels, all_probs) if len(np.unique(all_labels)) > 1 else 0

            # Store validation results
            all_val_losses.append(avg_val_loss)
            all_val_metrics.append({
                'accuracy': val_accuracy,
                'precision': val_precision,
                'auc_roc': val_auc_roc
            })

            # Update epoch progress bar with key metrics
            epoch_pbar.set_postfix(train_loss=f"{avg_train_loss:.4f}", val_loss=f"{avg_val_loss:.4f}")

            # Early stopping check (stop if validation loss stops improving)
            if early_stopping_patience and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
            elif early_stopping_patience:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping triggered after {epoch} epochs")
                    break

        # Report comprehensive metrics periodically
        if epoch % report_every == 0:
            print(f"Epoch {epoch}/{n_epoch} ({epoch/n_epoch:.0%}):")
            print(f"  Train loss: {avg_train_loss:.6f}")
            if validation_data:
                print(f"  Validation loss: {avg_val_loss:.6f}")
                print(f"  Validation accuracy: {val_accuracy:.4f}")
                print(f"  Validation AUC-ROC: {val_auc_roc:.4f}")

    # Prepare and return training results
    results = {
        'train_loss': all_train_losses,
    }

    if validation_data:
        results.update({
            'val_loss': all_val_losses,
            'val_metrics': all_val_metrics
        })

    return results