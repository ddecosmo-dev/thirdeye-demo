Overview of service architecture for cloud device.

This service will be run on either cloud or a local PC to send and recieve messages from the edge device and perform final cloud inference for a user. 
We are mainly concerned with constructing the backend in a manner such that adding a GUI is simple and highly modifiable. This should be a micro monolith, deployability and functionaility are the main focus

Services:
1. Frontend service: sends requests from the GUI including to start inference cycle on edge or to display images from the DB
2. Image Router: service recieves .zip or compressed image files w/ tags from edge service. Needs to verify integrity of files, unzip, and save to image storage.
Should this be a db or directory? It will include tags like image scores and what filters it passed
3. Cloud inference service: this will hold the .py files to do inference with mulitple models to get more refined scores, then to cluster and select the champions from an inference cycle. This service should hear from the image router when upload has been successful, then query the DB or storage directory to get the images from the relevant cycle, perform inference, and update the tags as needed. Once this is done, it should also send a message to the frontend service and GUI to allow the images to be displayed.

All services should have necessary safety checks while still being lightweight enough for a Pi Zero 2 W. The two external services need the most checks to ensure messages are recieved and images are transmitted correctly.

See diagram.png for a visual diagram of my thoughts on the architecture.

Other questions
I thought python flask was good for performance and readbaility, is this a good call
How should we organize data for each inference step? Actual db or just a temp directory?
How can we safely clean data, ensuring it was correctly sent before doing so

