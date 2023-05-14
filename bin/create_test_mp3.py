import torch
from klay_beam.transforms import numpy_to_mp3

def run():
    # create several seconds of random noise with a fade in
    sr = 44100
    seconds = 2
    total_samples = int(sr * seconds)
    audio_tensor = torch.rand(total_samples) * 2 - 1

    fade_in = torch.linspace(0., 1., total_samples) ** 2
    audio_tensor *= fade_in
    audio_tensor = audio_tensor.reshape(total_samples, 1)
    
    mp3_buffer = numpy_to_mp3(audio_tensor.numpy(), sr, noisy=True)

    with open("test.mp3", "wb") as out_file:
        out_file.write(mp3_buffer.getvalue())
        mp3_buffer.seek(0)

if __name__ == "__main__":
    run()
