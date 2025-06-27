import apache_beam as beam

from job_stats.transforms import AudioStats


def test_pipeline():
    with beam.Pipeline() as pipeline:
        # Assume 'audio_stats_pcoll' is the PCollection of AudioStats objects
        audio_stats_pcoll = pipeline | "Create AudioStats" >> beam.Create(
            [
                AudioStats(duration=3.5, is_stereo=True, sr=44100),
                AudioStats(duration=2.0, is_stereo=False, sr=48000),
                AudioStats(duration=4.5, is_stereo=True, sr=44100),
                AudioStats(duration=5.0, is_stereo=False, sr=96000),
            ]
        )

        # Count the total number of AudioStats objects
        total_count = (
            audio_stats_pcoll | "Count AudioStats" >> beam.combiners.Count.Globally()
        )

        # Sum the total duration of all AudioStats objects
        total_duration = (
            audio_stats_pcoll
            | "Extract Durations" >> beam.Map(lambda x: x.duration)
            | "Sum Durations" >> beam.CombineGlobally(sum)
        )

        # Count the number of stereo and mono audio
        stereo_count = (
            audio_stats_pcoll
            | "Extract Stereo Count"
            >> beam.Map(lambda x: ("stereo" if x.is_stereo else "mono", 1))
            | "Count Stereo and Mono" >> beam.combiners.Count.PerKey()
        )

        # Sum the sampling rates
        total_sr = (
            audio_stats_pcoll
            | "Extract Sampling Rates" >> beam.Map(lambda x: x.sr)
            | "Sum Sampling Rates" >> beam.CombineGlobally(sum)
        )

        # Output the results
        total_count | "Print Total Count" >> beam.Map(
            lambda x: print(f"Total Count: {x}")
        )
        total_duration | "Print Total Duration" >> beam.Map(
            lambda x: print(f"Total Duration: {x}")
        )
        stereo_count | "Print Stereo Count" >> beam.Map(print)
        total_sr | "Print Total SR" >> beam.Map(lambda x: print(f"Total SR: {x}"))
