import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import argparse 
import time 
os.environ['QT_QPA_PLATFORM'] = 'offscreen'  # Set the platform to offscreen to avoid GUI errors

# argument parser
parser = argparse.ArgumentParser(description='Inference on video')
parser.add_argument('--video', type=str, help='Path to the video file')
parser.add_argument('--confidence_threshold', type=float, default=0.8, help='Confidence threshold for inference')
args = parser.parse_args()  

# Define the classes
gun = {1: 'Machine_Gun', 2: 'HandGun'}

# Define the number of classes
num_classes = 3  # Change this to match your fine-tuned model's number of classes

# Load the pretrained model
model = fasterrcnn_resnet50_fpn(pretrained=True)

# Modify the box_predictor to match the fine-tuned model
in_features = model.roi_heads.box_predictor.cls_score.in_features
box_predictor = FastRCNNPredictor(in_features, num_classes)

# Set the modified box_predictor to the model
model.roi_heads.box_predictor = box_predictor

# Load the fine-tuned weights
model.load_state_dict(torch.load('fasterrcnn_resnet50_finetuned_fpn.pth'))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model.to(device)

model.eval()

def infer_video(video, confidence_threshold=0.8):
    # load the video
    cap = cv2.VideoCapture(video)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    out = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))

    while cap.isOpened():
        # start the timer
        start_time = cv2.getTickCount()
        ret, image = cap.read()
        if not ret:
            break
        original_image = image.copy()


        # Convert the image from OpenCV BGR format to PyTorch RGB format
        image = image[:, :, [2, 1, 0]]
        image = torch.from_numpy(image).permute(2, 0, 1)
        image = image.float() / 255.0


        # Put the image on the GPU if available
        image = image.to(device)

        # Add a batch dimension
        image = image.unsqueeze(0)
        
        # start time
        start_time_inference = time.time()
        # Perform inference
        with torch.no_grad():
            prediction = model(image)
        
        end_time_inference = time.time()
        print('Inference time: {:.4f}'.format(end_time_inference - start_time_inference))
        print('Inference speed: {:.2f} FPS'.format(1 / (end_time_inference - start_time_inference)))
        cv2.putText(original_image, 'Inference time: {:.4f}s'.format(end_time_inference - start_time_inference), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(original_image, 'Inference speed: {:.2f} FPS'.format(1 / (end_time_inference - start_time_inference)), (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        
        print(prediction[0])
        # Filter out the predictions with low confidence
        for element in prediction[0]['scores']:
            score = element.cpu().numpy()
            if score < confidence_threshold:
                break

            # Draw the bounding boxes on the image
            for element in prediction[0]['boxes']:
                box = element.cpu().numpy()
                cv2.rectangle(original_image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (255, 0, 0), 2)
                cv2.putText(original_image, str(score), (int(box[0]), int(box[1]+40)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

            for element in prediction[0]['labels']:
                label = element.cpu().numpy()
                cv2.putText(original_image, str(gun[int(label)]), (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - start_time)
        cv2.putText(original_image, 'Final FPS: {:.2f}'.format(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.imshow('image', original_image)

        #DOWNLAOD THE VIDEO
        out.write(original_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':
    #parse the arguments
    infer_video(args.video, args.confidence_threshold)
