FROM klaymusic/klay-beam:0.13.7-py3.9-beam2.53.0-torch2.0

WORKDIR /klay/app

# Add all the model files from essentia
ADD https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/approachability/approachability_2c-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/engagement/engagement_2c-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/nsynth_instrument/nsynth_instrument-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/nsynth_reverb/nsynth_reverb-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/classification-heads/mtt/mtt-discogs-effnet-1.pb ./models/
ADD https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb ./models/
ADD https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.pb ./models/


COPY ./jobs/job_essentia ./job_essentia
RUN python3 -m pip install './job_essentia'
RUN pip check

# Download the models
RUN python3 -c "import job_essentia.transforms as t; t.construct_essentia_classifiers()"


ENTRYPOINT ["/opt/apache/beam/boot"]
