# MuJoCo Simulation of Cart-pole System
This repo consist of the simulation files to model and simulate a cart-pole system using 3 controllers - PID, Pole placement and LQR.
## File Descriptions
### model.xml
This is the XML file used to model the cart-pole system. It consists of a cart weighing 1Kg and a pole weighing 0.1 Kg. Additional railings are present for visualization and they do not interact with the cart-pole.

### PID_cart_pole.py
This file contains the PID controller for cart-pole system. A cascaded PID structure is implemented to regulate the pole angle and cart position. Performance metrics like Peak Force, Control effort, Settling time are evaluated in the end.

### PP_cart_pole.py
This file contains the state space modelling with pole placement for the cart-pole system. The A and B matrices contain hard-coded values. Given the required pole locations, the controller calculates the gain vector and applies a control input u = -KX.

### LQR_cart_pole.py
This file contains the state space modelling and LQR for the cart-pole system. The A and B matrices contain hard-coded values. Given the Q and R matrices, the controller calculates the gain vector by solving the continuous time Riccati equation and applies a control input u = -KX.
