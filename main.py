import torch
from scripts.data_processing import EagleEyesDataset, collate_fn, split_dataset, visualize_annotations
from scripts.train import train_model
from scripts.evaluate_model import evaluate_model  # <-- Now imported here
from model.my_model import get_model
import os

from scripts.cleanup_generated_data import clean_generated_folders
clean_generated_folders()

def main():
    # Paths and parameters
    json_path = "./data/db_cache.json"
    image_dir = "./data/images"
    train_image_dir = "./data/data_processing/train/image"
    train_label_dir = "./data/data_processing/train/labels"
    test_image_dir = "./data/data_processing/test/image"
    test_label_dir = "./data/data_processing/test/labels"
    viz_dir = "./data/data_processing/visualized_images"
    output_dir = "./forecast_images"
    model_save_path = "./model/model.pth"
    num_classes = 2
    num_epochs = 20
    batch_size = 2
    patience = 5
    train_ratio = 0.8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Step 1: Split dataset
    train_image_dir, train_label_path, test_image_dir, test_label_path = split_dataset(
        json_path, image_dir, train_image_dir, train_label_dir, test_image_dir, test_label_dir, train_ratio=train_ratio
    )

    # Step 2: Optional annotation visualization
    visualize_annotations(train_image_dir, train_label_path, viz_dir)
    visualize_annotations(test_image_dir, test_label_path, viz_dir)

    # Step 3: Load datasets
    train_dataset = EagleEyesDataset(train_image_dir, train_label_path)
    test_dataset = EagleEyesDataset(test_image_dir, test_label_path)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Step 4: Initialize and train model
    model = get_model(num_classes)
    model.to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.005, momentum=0.9, weight_decay=0.0005)
    train_model(model, train_loader, optimizer, device, num_epochs, model_save_path, patience)
    model.load_state_dict(torch.load(model_save_path, map_location=device)) #Reload model weights for evaluation

    # Step 5: Evaluate model
    pr_auc = evaluate_model(model, test_loader, device, output_dir, score_threshold=0.3)
    print(f"Test PR-AUC: {pr_auc:.4f}")

if __name__ == "__main__":
    main()
