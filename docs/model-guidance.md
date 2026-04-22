Current goal, redesign the edge device service. 
Overall goals, 
facilitate easy deployment and testing of models to Luxonis camera and store images with related information.

i.e unique image name
time of capture
passing tag. i.e passed prefilter, passed model, or failed either. Can be an int or something else
Edge device model score: if applicable.


Actual implementation, 2 REST services. 
For now, assume they are two services within the same deployment environment, designed to run as lightweight as possible on a raspberry pi zero 2 W.

Service 1: Coordinator

External endpoint to localhost, and later cloud service.
Takes inputs to being inference. Endpoint should contain tags for the length of time to run in seconds, plus an optional run name tag.

External endpoint should also be able to recieve an abort message from localhost/cloud. This can automatically shutdown the services if needed.

The cooridinator should also communicate with the external host, sending logs to make the process readable during debugging.

At startup, the coordinator will initialize and load the pipeline.py required for the DepthAI V3 to get the model started on the luxonis Oak D Lite camera being used. The coordinator will also tell the image processor service to start at this time. When the abort is called, or inference is over. The coordinator will tell the Luxonis camera to stop (if this is possible) and tell the image processor service to finish any images in its queue and proceed with its final sequence. 

There should be robust logging and error handling to improve debugging during local testing.

Luxonis Model Pipeline:
To load the model onto the Luxonis camera we are going to use models using the DepthAI V3 system. a model blob file is already provided and the desired workflow is below.


1. Get an image from the camera at a set sampling rate (i.e 1 FPS)
2. Save the full image or an acceptable JPEG or whatever for storage
3. downsample the image to 320 x 240
4. Pass the downsampled image to the prefilter, if it passes proceed. Else save the full jpeg along with metadata.
5. After prefilter, OPTINAL STEP! the current model used is based on dino, may require normalization to do accurate inference. Check if model distiller actually includes it as part of the blob file or if it needs to be part of the inference process
6. Pass downsampled image to blob model, complete inference and save score to metadata
7. Check inference score to threhold, generally 0.5.Update metadata as needed.
i.e if above, save as passing edge, 
8. Send the jpeg image to the image processor alongside associated metadata


Service 2: Image Processor

Once told to start by the coordinator, the image processor will listen for images from the Oak D camera. When images are recieved, they should be stored and placed in a temporary directory. Associated metadata should be stored in a document within this temp directory. This could be a json, csv or whatever else to make this as smooth as possible. 

This will continue until the process is killed or the image processor recieves the stop command from the coordinator for an abort or time-elapsed message. 

At this point, the image processor will finish processing any messages, then zip the temporary files into a compact zip file for easy transfer. Once this is confirmed the temporary directory can be removed.

After edge testing, a new feature will be added. The .zip file will be routed to the cloud service through a new external endpoint, completing the loop. Do not do this yet.


