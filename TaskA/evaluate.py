import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def get_val_loader(val_dir, batch_size, num_workers=4):
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return val_loader, val_dataset.classes


def evaluate_metrics(model, loader, device, class_names):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            preds = (torch.sigmoid(outputs) > 0.5).long().cpu().view(-1).tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds)
    rec = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    print(f"Accuracy: {acc:.2f}%")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Evaluate saved ConvNeXt model")
    parser.add_argument('--val-dir', type=str, required=True)
    parser.add_argument('--model-path', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_loader, class_names = get_val_loader(args.val_dir, args.batch_size)

    model = models.convnext_base(pretrained=False)
    model.classifier[2] = nn.Linear(model.classifier[2].in_features, 1)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)

    evaluate_metrics(model, val_loader, device, class_names)

if __name__ == '__main__':
    main()
