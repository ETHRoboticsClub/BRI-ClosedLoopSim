"""Run the same sensing and encoding logic for each fingertip."""
#we can also adjust the sensing/ encoding logic uniquely for each fingertip as we test them

from .sensor import ForceSensor, SensorConfig
from .encoding import AmplitudeEncoder, EncodingConfig

#here we can adjust how many sensors, this class updates the values with each step
class FeedbackPipeline:
    def __init__(self, n_sensors=4, sensor_config=None, encoding_config=None):
        if n_sensors < 1:
            raise ValueError("n_sensors must be positive")
        self.sensors = [ForceSensor(sensor_config) for _ in range(n_sensors)]
        self.encoders = [AmplitudeEncoder(encoding_config) for _ in range(n_sensors)]

    def update(self, forces_n):
        """Return (sensor readings, vibration intensity commands).

        forces_n must contain one nonnegative force in newtons per sensor.
        Contact points belonging to the same pad must be grouped first.
        """
        if len(forces_n) != len(self.sensors):
            raise ValueError(f"Expected {len(self.sensors)} forces, got {len(forces_n)}")

        readings = [
            sensor.update(force)
            for sensor, force in zip(self.sensors, forces_n)
        ]
        commands = [
            encoder.update(reading.normalized)
            for encoder, reading in zip(self.encoders, readings)
        ]
        return readings, commands

    def reset(self):
        for sensor in self.sensors:
            sensor.reset()
        for encoder in self.encoders:
            encoder.reset()
