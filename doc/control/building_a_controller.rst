###################################################
Building a Controller
###################################################

The following documents the development of a new controller.
In this case we are going to implement an arbitrary controllable storage unit. This
may be a battery, an electrically powered car or some sort of reservoir storage.


Parent-Class
---------------
First we start by creating a new file *control/storage_control.py*, containing our new class.

::

    import control.basic_controller

    class Storage(control.basic_controller.Controller):
        """
            Example class of a Storage-Controller. Models an abstract energy storage.
        """

        def __init__(self):
            # init object here
            pass

        def time_step(self, time):
            """
            Note: This method is ONLY being called during time-series simulation!

            It is the first call in each time step, thus suited for things like
            reading profiles or prepare the controller for the next control step.
            """
            pass

        def write_to_net(self):
            """
            This method will write any values the controller is in charge of to the
            data structure. It will be called at the beginning of each simulated
            loadflow, in order to ensure consistency between controller and
            data structure.

            You will probably want to write the final state of the controller to the
            data structure at the end of the control_step using this method.
            """
            pass

        def initialize_control(self):
            """
            Some controller require extended initialization in respect to the
            current state of the net (or their view of it). This method is being
            called after an initial loadflow but BEFORE any control strategies are
            being applied.

            This method may be interesting if you are aiming for a global
            controller or if it has to be aware of its initial state.
            """
            pass

        def is_converged(self):
            """
            This method calculated whether or not the controller converged. This is
            where any target values are being calculated and compared to the actual
            measurements. Returns convergence of the controller.
            """
            return True

        def control_step(self):
            """
            If the is_converged method returns false, the control_step will be
            called. In other words: if the controller did not converge yet, this
            method should implement actions that promote convergence e.g. adapting
            actuating variables and writing them back to the data structure.

            Note: You might want to store the mismatch calculated in is_converged so
            you don't have to do it again. Also, you might want to write the
            reaction back to the data structure (use write_to_net).
            """
            pass

        def finalize_step(self):
            """
            Note: This method is ONLY being called during time-series simulation!

            After each time step, this method is being called to clean things up or
            similar. The OutputWriter is a class specifically designed to store
            results of the loadflow. If the ControlHandler.output_writer got an
            instance of this class, it will be called before the finalize step.
            """
            pass

.. note::
       Import and inherent from the parent class :mod:`Controller` and override methods you
       would like to use. Also remember that *is_converged()* returns the boolean value of
       convergence.

Next we write the actual code for the methods. We choose to represent the storage-unit as a static
generator in pandapower. To do so we overwrite *__init__* and initiate all the attributes of our
class with the values of the corresponding generator using its ID.

::

    def __init__(self, net, gid, soc, capacity, sizing):

        # read generator attributes from net
        self.gid = gid
        self.bus = net.sgen.at[gid, "bus"]
        self.p_kw = net.sgen.at[gid, "p_kw"]
        self.q_kvar = net.sgen.at[gid, "q_kvar"]
        self.sn_kva = net.sgen.at[gid, "sn_kva"]
        self.name = net.sgen.at[gid, "name"]
        self.gen_type = net.sgen.at[gid, "type"]
        self.in_service = net.sgen.at[gid, "in_service"]

        #specific attributes
        self.capacity = capacity
        self.soc = soc
        self.sizing = sizing

Methods that should be shared amongst all storage classes have to be implemented here as well. ::

    def get_stored_ernergy(self):
        # do some "complex" calculations
        return self.capacity * self.soc

After doing so, our parent class is finished. But now
that we have a parent class, lets actually use it by implementing a
subclass of it. In this example it will be a simple battery.

Child-Class
--------------------
Again create a new file *control/storage/electric_car.py* for our new :mod:`ECar` class. Note: It is a good
idea to keep your project files organized by creating subfolders for closely related classes
or scripts.

::

    import control.controller.storage_control

    class Battery(control.controller.storage_control.Storage):
        """
        Models a battery plus inverter.
        """

        def __init__(self):
            # init object here
            pass

        def time_step(self, time):
            # change state according to profile
            pass

        def write_to_net(self):
            # write current P and Q values to the data structure
            pass

        def is_converged(self):
            # calculate convergence criteria
            pass

        def control_step(self):
            # apply control strategy
            return True


Except the import and its inherence, this class looks quite the same.
We want to make some adjustments though:

::

    def __init__(self, net, gid, soc, capacity, sizing, p_profile=None, data_source=None):
        super(Battery, self).__init__(net, gid, soc, capacity, sizing)

        # profile attributes
        self.data_source = data_source
        self.p_profile = p_profile
        self.last_time_step = None

Lets have a closer look at this code. We can call the constructor of
the parent class letting it handle all the parameters and set attributes by using the super
mechanism: ``super(CHILD-CLASS, self).__init__()``. Additionally we want read values from a profile. 

.. note::
    If you strictly follow the order of parameters the parents constructor expects,
    you can refrain from writing ``net=net`` and go with
    ``super(Battery, self).__init__(net, gid, soc, capacity, sizing)`` instead.

As a first step we want our controller to be able to write its P and Q values back to the
data structure.

::

    def write_to_net(self):
        # write p, q to bus within the net
        self.net.sgen.at[self.gid, "p_kw"] = self.p_kw
        self.net.sgen.at[self.gid, "q_kvar"] = self.q_kvar

::

    def is_converged(self):
        # calculate if controller is converged
        is_converged = "some boolean logic"

        return bool(is_converged)

In case the controller is not yet converged, the control step is executed. In the example it simply
adopts a new value according to the previously calculated target and writes back to the net.

::

    def control_step(self):
        # some control mechanism
        
        # write p, q to bus within the net
        self.write_to_net()

In a time-series simulation the battery should read new power values from a profile and keep track
of its state of charge as depicted below.

::

    def time_step(self, time):
        # keep track of the soc (assuming time is given in seconds)
        if self.last_time_step is not None:
            self.soc += self.capacity / (self.p_kw * (self.current_time_step-self.last_time_step) / 3600)
        self.last_time_step = time

        # read new values from a profile
        if self.data_source:
            if self.p_profile:
                self.p_kw = self.data_source.get_time_step_value(time_step=time,
                                                                profile_name=self.p_profile)

We are now ready to create objects of our newly implemented class and simulate with it!

.. note::
    Decent commentary is best practice. It is very handy for people reviewing
    your code or in case you want to look into the code a few months after
    implementation.