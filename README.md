# mlflow-practice notebook

## How to run this example
1. Install Docker and MLFlow (pip install mlflow)

2. Build the docker image `docker build -t mlflow-example-docker -f Dockerfile .`

3. Run the project with `mlflow run https://github.com/marianore-beepboop/mlflow-practice.git -P alpha=0.5`

## How to run with docker-compose
1. From `docker` directory run `docker-compose --env-file .env up --build` to build and create the Docker Compose with a Mlflow workspace
container and the Mlflow Server/UI needed to track and modify your
Mlflow Projects.

2. You can work from your preferred IDE entering your workspace container with the following command:
`ssh -p 22 dockeruser@localhost`
If asked for a password type `123`.
Username, password and port can easily be modified in the docker-compose file if you like.

3. From there you can easily work in your environment with Mlflow and its dependencies. Start by typing `conda init [preferred-terminal]` and then `conda activate`. Close and reopen a terminal and then try running `which python` and `python --version` to check that you're running from the conda environment. If you can run `mlflow --version` you are good to go.

4. Done! Now you can run your ML models within the Docker container and check the Mlflow UI in your own PC at `localhost:5000` for Model Management and Experimentation, all from an isolated Docker container with all the features that you need.
