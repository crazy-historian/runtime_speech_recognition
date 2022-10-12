import wandb


if __name__ == "__main__":
    dir_name = '/media/maxim/Programming/voice_datasets/timit/TIMIT_2/data'
    wandb.login(key='')
    with wandb.init(project='phoneme_recognizer', job_type='dataset') as run:
        artifact = wandb.Artifact(name='timit-dataset', type='dataset')
        artifact.add_dir(dir_name, name='arctic')
        run.log_artifact(artifact)
