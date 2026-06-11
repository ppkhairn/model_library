import argparse
import os
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as T

# Our locked-down list of supported models
AVAILABLE_MODELS = [
    "resnet50",
    "efficientnet_b3",
    "vit_b_16",
    "yolov8",
    "faster_rcnn"
]

def load_selected_model(model_name):
    """Loads the off-the-shelf model based on your choice."""
    print(f"-> Loading pre-trained weights for {model_name}...")
    
    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        categories = weights.meta["categories"]
        return model.eval(), categories, "classification"

    elif model_name == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT
        model = models.efficientnet_b3(weights=weights)
        categories = weights.meta["categories"]
        return model.eval(), categories, "classification"

    elif model_name == "vit_b_16":
        weights = models.ViT_B_16_Weights.DEFAULT
        model = models.vit_b_16(weights=weights)
        categories = weights.meta["categories"]
        return model.eval(), categories, "classification"

    elif model_name == "faster_rcnn":
        weights = models.detection.FasterRCNN_MobileNet_V3_Large_320_FPN_Weights.DEFAULT
        model = models.detection.fasterrcnn_mobilenet_v3_large_320_fpn(weights=weights)
        categories = weights.meta["categories"]
        return model.eval(), categories, "detection"

    elif model_name == "yolov8":
        from ultralytics import YOLO
        model = YOLO("yolov8n.pt")
        return model, None, "yolo"

def run_classification_inference(model, image, categories):
    """Handles inference for ResNet, EfficientNet, and ViT."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted_idx = torch.max(probabilities, dim=0)
    
    return {
        "prediction": categories[predicted_idx.item()],
        "confidence": round(float(confidence.item()), 4)
    }

def run_detection_inference(model, image, categories):
    """Handles inference for Faster R-CNN and annotates the image."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    transform = T.ToTensor()
    input_tensor = transform(image_rgb).unsqueeze(0)

    with torch.no_grad():
        predictions = model(input_tensor)[0]
    
    # Create a copy of the image to draw on
    annotated_image = image.copy()
    
    detected = []
    for score, box, label in zip(predictions["scores"], predictions["boxes"], predictions["labels"]):
        if score >= 0.5:
            box_xyxy = [int(coord) for coord in box.tolist()]
            obj_name = categories[label.item()]
            conf_val = round(float(score.item()), 4)
            
            detected.append({
                "object": obj_name,
                "confidence": conf_val,
                "box_xyxy": box_xyxy
            })
            
            # --- Draw Bounding Box ---
            # Green rectangle (BGR: 0, 255, 0), thickness of 2 pixels
            cv2.rectangle(annotated_image, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (0, 255, 0), 2)
            # Text label: "label: confidence"
            label_text = f"{obj_name}: {conf_val}"
            cv2.putText(annotated_image, label_text, (box_xyxy[0], max(15, box_xyxy[1] - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
    return {"detected_objects": detected}, annotated_image

def main():
    parser = argparse.ArgumentParser(description="Image Inference CLI")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, required=True, choices=AVAILABLE_MODELS, help="Model name from the list")
    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"Error: Image path '{args.image}' does not exist.")
        return
    image = cv2.imread(args.image)

    model, categories, task_type = load_selected_model(args.model)

    print("Running prediction...")
    annotated_image = None
    
    if task_type == "classification":
        result = run_classification_inference(model, image, categories)
        
    elif task_type == "detection":
        result, annotated_image = run_detection_inference(model, image, categories)
        
    elif task_type == "yolo":
        results = model(args.image, verbose=False)[0]
        result = {"detected_objects": []}
        annotated_image = image.copy()
        
        for box in results.boxes:
            if box.conf[0] >= 0.5:
                box_xyxy = [int(coord) for coord in box.xyxy[0].tolist()]
                obj_name = model.names[int(box.cls[0])]
                conf_val = round(float(box.conf[0]), 4)
                
                result["detected_objects"].append({
                    "object": obj_name,
                    "confidence": conf_val,
                    "box_xyxy": box_xyxy
                })
                
                # --- Draw Bounding Box for YOLO ---
                cv2.rectangle(annotated_image, (box_xyxy[0], box_xyxy[1]), (box_xyxy[2], box_xyxy[3]), (0, 255, 0), 2)
                label_text = f"{obj_name}: {conf_val}"
                cv2.putText(annotated_image, label_text, (box_xyxy[0], max(15, box_xyxy[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 4. Output text results
    print("\n--- Inference Result ---")
    import json
    print(json.dumps(result, indent=4))

    # 5. Save the image if bounding boxes were drawn
    if annotated_image is not None:
        output_path = "output_prediction.jpg"
        cv2.imwrite(output_path, annotated_image)
        print(f"\n[Success] Bounding boxes drawn! Saved annotated image to: {output_path}")

if __name__ == "__main__":
    main()