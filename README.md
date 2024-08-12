# AIdventure_Server
The server on which connects AIdventure game.

## How to install AIdventure_Server on a linux server

1. Clone the repository `git clone git@github.com:LyaaaaaGames/AIdventure_Server.git`
2. Run the install.sh `./install.sh`. The installer will ask if you want to support CUDA.
3. Run the run.sh `./run.sh`

You might have to give the scripts the executable permission with `chmod +x script_name`

## How to update ?

1. Download the new version
2. Overwrite with the new files
3. Run `update_env.sh` to update conda's environment (It might do nothing if the environment didn't change).

## How to edit the server config

1. Open `server/config.py`
2. Edit what you want.

## More info

- [Code of conduct](https://github.com/LyaaaaaGames/AIdventure_Server/blob/main/CODE_OF_CONDUCT.md)
- [Contributing](https://github.com/LyaaaaaGames/AIdventure_Server/blob/main/CONTRIBUTING.md)
- [Style guide](https://github.com/LyaaaaaGames/AIdventure_Server/blob/main/style_guide.md)
