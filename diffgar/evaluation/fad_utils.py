from .fad import FrechetAudioDistance


class Hyperparameters:
    def __init__(self):
        self.fad_models = ['vggish', 'clap']
        self.fad_background_embeddings = [
            'background_embeddings/vggish_background_embeddings.npz',
            'background_embeddings/clap_background_embeddings.npz'
        ]
        self.fad_workers = 4
        self.data_path_test = 'data/test_data/'
    

def compute_fad(fad_path):
    
    hparams = Hyperparameters()
    scores = []
    for i,model_name in enumerate(hparams.fad_models):

        if model_name == 'vggish':
            frechet = FrechetAudioDistance(
                model_name="vggish",
                sample_rate=16000,
                use_pca=False, 
                use_activation=False,
                verbose=True,
                audio_load_worker=hparams.fad_workers)
        elif model_name == 'clap':
            frechet = FrechetAudioDistance(
                model_name="clap",
                sample_rate=48000,
                submodel_name="630k-audioset",
                verbose=True,
                enable_fusion=False,
                audio_load_worker=hparams.fad_workers)
        else:
            raise NameError('Must be (vggish, clap)')

        score = frechet.score(
            hparams.data_path_test,
            fad_path,
            background_embds_path=hparams.fad_background_embeddings[i],
            dtype="float32")
        scores.append(score)
        
    return scores