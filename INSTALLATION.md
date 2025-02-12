# Install from Source
* Install git
* Use conda or mamba
* Use vscode
* Clone project
* Create env
~~~bash
~$ conda create --prefix envs/python3.8 python=3.8 pip
or
~$ mamba create --prefix envs/python3.8 python=3.8 pip
~~~
* Install pytorch
~~~bash
~$ conda install pytorch::pytorch torchvision torchaudio -c pytorch
or 
~$ mamba install pytorch::pytorch torchvision torchaudio -c pytorch
~~~
* Install other dependacies
~~~bash
~$ python3 -m pip install -r requirements/tequirements.dev.txt
~~~
* Work!
