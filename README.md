# IPaLot: an Intelligent Parking Lot

This project describes a control system for a parking lot, 
based on reinforcement learning, where an AI can take control
of the customer's car and dispach/retrieve it to/from a
designated parking spot.

Here you can find the training simulation, working on pygame.
The training method used is a multi-agent version of the [A3C model](https://arxiv.org/pdf/1602.01783.pdf),
based on the [implementation](https://jaromiru.com/2017/03/26/lets-make-an-a3c-implementation/) by Jaromír Janisch
avalaible [here](https://github.com/jaara/AI-blog/blob/master/CartPole-A3C.py).
A pretraining program, based on an A\* search, is also included: it finds optimal paths to/from parking spots for a car,
and uses the accumulated experience to train the same neural network used in the main program.

Check the linked images here for a description of the
[threaded](https://github.com/orla84/IPaLot/blob/master/images/threads.png)
structure of the code and of the architecture of the 
[neural network](https://github.com/orla84/IPaLot/blob/master/images/Brain.png)

#### Installing (conda virtual environment)

* conda create --name myenv python=3.6.3
* conda activate myenv
* pip install -r requirements.txt

#### Running

* **python A_star_pretrain.py**  -- runs the pretraing based on A\* search
* **python main.py**  --  runs the A3C training simulation ( hyperparameters tuning can be performed changing the variables in cfg.py)

![game_image](https://github.com/orla84/IPaLot/blob/master/images/a3c_4cars.PNG)