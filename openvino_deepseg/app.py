import argparse
import cv2
import sys
import numpy as np
import subprocess as sp
from inference import Network

INPUT_STREAM = "videos/smile_pal.mp4"
CPU_EXTENSION = "/media/anilsathyan7/work/vino/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"#"/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

# Ffmpeg command for encoding output video
command = [ 'ffmpeg',
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', '1024x576', 
        '-pix_fmt', 'bgr24',
        '-re',
        '-i', '-', 
        '-r', '15',
        '-an', 
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-f', 'flv',
        'rtmp://0.0.0.0:1935/LiveApp/segme live=1' ]

# Pipe the model output to server
pipe = sp.Popen( command, stdin=sp.PIPE, stderr=sp.PIPE)

def get_args():
    '''
    Gets the arguments from the command line.
    '''
    parser = argparse.ArgumentParser("Run inference on an input video")
    # -- Create the descriptions for the commands
    m_desc = "The location of the model XML file"
    i_desc = "The location of the input file"
    d_desc = "The device name, if not 'CPU'"

    # -- Add required and optional groups
    parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')

    # -- Create the arguments
    required.add_argument("-m", help=m_desc, required=True)
    optional.add_argument("-i", help=i_desc, default=INPUT_STREAM)
    optional.add_argument("-d", help=d_desc, default='CPU')
    args = parser.parse_args()

    return args


def infer_on_video(args):
    # Initialize the Inference Engine
    plugin = Network()

    # Load the network model into the IE
    plugin.load_model(args.m, args.d, CPU_EXTENSION)
    net_input_shape = plugin.get_input_shape()

    # Get and open video capture
    cap = cv2.VideoCapture(args.i)
    cap.open(args.i)
    
    # Grab the shape of the input 
    width = int(cap.get(3))
    height = int(cap.get(4))

    # Create a video writer for the output video
    fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
    out = cv2.VideoWriter('out.mp4', fourcc, 15, (width,height))
    frame_count = 0
    

    # Process frames until the video ends, or process is exited
    while cap.isOpened():
        # Read the next frame
        flag, input_frame = cap.read()
        
        if not flag:
            break
        key_pressed=cv2.waitKey(60)        

        # Pre-process the frame
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        p_frame = cv2.resize(input_frame, (net_input_shape[3], net_input_shape[2]))
        p_frame = p_frame.transpose((2,0,1))
        p_frame = p_frame.reshape(1, *p_frame.shape)
        
        
        # Perform inference on the frame
        plugin.async_inference(p_frame)

        # Get the output of inference
        if plugin.wait() == 0:
            result = plugin.extract_output()
            result = result.transpose((1,2,0))

            # Get semantic mask for person class
            person_mask = np.uint8(result==15)
            person_mask = np.dstack([person_mask,person_mask,person_mask])
            person_mask = cv2.resize(person_mask, (width, height))
            
            # Create the overlay mask
            overlay = np.zeros_like(person_mask)
            overlay[:] = (127, 0, 0) 
                    
            # Add overlay-mask over input frame
            overlay_mask = person_mask * overlay
            assert person_mask.shape==overlay_mask.shape, "Raw person mask and overlay mask should be of same dimensions"
            output_frame = cv2.addWeighted(input_frame, 1, overlay_mask, 0.9, 0)
            
            # Write output frames to video
            output_frame=cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            out.write(output_frame)

            # Pipe output frames to server
            pipe.stdin.write( output_frame.tostring() )

            # Show frame counter
            frame_count=frame_count+1
            print('Frame count: '+str(frame_count))
          
        # Break on keyboard interrupt
        if key_pressed == 27:
            break

    # Release the capture and destroy any OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    


def main():
    args = get_args()
    infer_on_video(args)


if __name__ == "__main__":
    main()
