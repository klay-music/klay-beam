#!/bin/bash

# Prepare models/ directory
mkdir -p models
cd models

# Download the DiscogsEffnet feature extractor and all the classifier heads
curl -LO https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mtg_jamendo_genre/mtg_jamendo_genre-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/approachability/approachability_2c-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/engagement/engagement_2c-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/danceability/danceability-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_aggressive/mood_aggressive-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_happy/mood_happy-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_party/mood_party-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_relaxed/mood_relaxed-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_sad/mood_sad-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mtg_jamendo_moodtheme/mtg_jamendo_moodtheme-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mtg_jamendo_instrument/mtg_jamendo_instrument-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_acoustic/mood_acoustic-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mood_electronic/mood_electronic-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/voice_instrumental/voice_instrumental-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/timbre/timbre-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/nsynth_instrument/nsynth_instrument-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/nsynth_reverb/nsynth_reverb-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/tonal_atonal/tonal_atonal-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mtg_jamendo_top50tags/mtg_jamendo_top50tags-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/mtt/mtt-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/tempo/tempocnn/deeptemp-k16-3.pb
curl -LO https://essentia.upf.edu/models/audio-event-recognition/yamnet/audioset-yamnet-1.pb