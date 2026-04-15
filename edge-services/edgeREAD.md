Overview of service architecture for edge device.

Need a lightweight service that runs on a Pi Zero 2 W for edge inference using an Oak Lite camera as the main compute for ML inference.
Should be containerized if possible, as long as this does not decrease performance. This should be a monolith, deployability and lightness is the focus

Services:
1. Should recieve REST messages to start, stop, or abort current edge inference.
2. When inference is started, the model should tell the Oak camera to use the pipeline attached, alongside model files and other nodes
3. When started, another service or structure should listen for images coming from the Oak, as they are recieved they should be routed to the storage for that inference cycle
4. When inference is done, defined by user input to service. The other services, other than the cloud listener, should stop. An image transmitter should then compress the images from the current inference cycle and transmit to the cloud access. 
Is this best done as an HTTP request or something else?
Is .zip best for this or something else?

All services should have necessary safety checks while still being lightweight enough for a Pi Zero 2 W. The two external services need the most checks to ensure messages are recieved and images are transmitted correctly.

See diagram.png for a visual diagram of my thoughts on the architecture.

Other questions
I thought python flask was good for performance and readbaility, is this a good call
How should we organize data for each inference step? Actual db or just a temp directory?
How can we safely clean data, ensuring it was correctly sent before doing so

