# DoomRL

To run code, set the `architecture` and `config_file` variables in `train.py` and run the command:

<pre>
python train.py
</pre>

Acceptable values for `architecture` are:
<ul>
<li> drqn
<li> c51
<li> a2c
</ul>

Model weights are saved in the models directory and the screen buffers of every thousandth game are saved in the gif directory.

The agents directory consists of modified code from [repo name] while the scenarios folder contains the doom game maps and configurations.

