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
~$ conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
or 
~$ mamba install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 pytorch-cuda=12.1 -c pytorch -c nvidia
~~~
* Install other dependacies
~~~bash
~$ python3 -m pip install -r requirements/requirements.dev.txt
~~~
* Work!
