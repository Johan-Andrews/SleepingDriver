"""Visual and escalating acoustic alert manager."""

from __future__ import annotations
import logging
import math
import os
import time
from typing import Optional, Tuple
import numpy as np

from src.decision.state_machine import SystemState

logger = logging.getLogger(__name__)


class SynthesizedBuzzer:
    """Generates and plays escalating buzzer tones using pygame.mixer and NumPy."""

    def __init__(self, sample_rate: int = 44100) -> None:
        self.sample_rate = sample_rate
        self.mixer_available = False
        self._init_mixer()

        # Cache synthesized sound chunks
        self._sound_low = None
        self._sound_med = None
        self._sound_high = None
        self._channel = None

        if self.mixer_available:
            self._generate_tones()

    def _init_mixer(self) -> None:
        """Initialize pygame mixer safely."""
        try:
            import pygame
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=self.sample_rate, size=-16, channels=1, buffer=512)
            self.mixer_available = True
            logger.info("Pygame audio mixer initialized.")
        except Exception as e:
            logger.warning("Audio mixer unavailable: %s. Audio alerts disabled.", e)
            self.mixer_available = False

    def _generate_tone_sound(self, freq: float, duration_s: float):
        """Create a pygame Sound object from a sine wave."""
        try:
            import pygame
            t = np.linspace(0, duration_s, int(self.sample_rate * duration_s), endpoint=False)
            waveform = 0.5 * np.sin(2 * np.pi * freq * t)
            # 16-bit signed PCM
            samples = (waveform * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(samples)
        except Exception as e:
            logger.debug("Could not generate tone sound: %s", e)
            return None

    def _generate_tones(self) -> None:
        """Pre-render warning tones."""
        self._sound_low = self._generate_tone_sound(500.0, 0.25)
        self._sound_med = self._generate_tone_sound(750.0, 0.15)
        self._sound_high = self._generate_tone_sound(1000.0, 0.10)

    def play_pattern(self, urgency: float, is_active: bool) -> None:
        """Play escalating sound depending on urgency (0.0 to 1.0)."""
        if not self.mixer_available or not is_active or urgency <= 0.0:
            return

        try:
            import pygame
            if self._channel is None:
                self._channel = pygame.mixer.Channel(0)

            if not self._channel.get_busy():
                if urgency < 0.4 and self._sound_low:
                    self._channel.play(self._sound_low)
                elif urgency < 0.8 and self._sound_med:
                    self._channel.play(self._sound_med)
                elif self._sound_high:
                    self._channel.play(self._sound_high)
        except Exception as e:
            logger.debug("Playback error: %s", e)

    def stop(self) -> None:
        """Silence any playing buzzer."""
        if self.mixer_available and self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass


class AlertManager:
    """Coordinates visual banner styles and audio buzzer alerts."""

    def __init__(self) -> None:
        self.buzzer = SynthesizedBuzzer()
        self.is_buzzer_active: bool = False

    def update(self, state: SystemState, urgency: float) -> Tuple[Tuple[int, int, int], str, bool]:
        """
        Update alert outputs for UI rendering and audio.
        Returns:
            color_rgb: Banner color
            display_text: Status banner string
            sound_playing: Buzzer active status
        """
        sound_active = False

        if state == SystemState.ALERT:
            color = (40, 167, 69)  # Green
            text = "SYSTEM ACTIVE - DRIVER ALERT"
            self.buzzer.stop()

        elif state == SystemState.CAUTION:
            color = (255, 193, 7)  # Amber / Yellow
            text = "CAUTION - ATTENTION DECREASING"
            self.buzzer.stop()

        elif state == SystemState.WARNING:
            color = (253, 126, 20)  # Orange
            text = f"WARNING! WAKE UP! ESCALATION: {int(urgency * 100)}%"
            sound_active = True
            self.buzzer.play_pattern(urgency, is_active=True)

        elif state == SystemState.INTERVENTION:
            color = (220, 53, 69)  # Red
            text = "EMERGENCY INTERVENTION - BRAKING ENGAGED"
            sound_active = True
            self.buzzer.play_pattern(1.0, is_active=True)

        elif state == SystemState.SAFE_STOP:
            color = (139, 0, 0)  # Dark Red
            text = "VEHICLE STOPPED - HAZARDS ACTIVE - CALLING HELP"
            sound_active = False
            self.buzzer.stop()

        elif state == SystemState.RECOVERY:
            color = (23, 162, 184)  # Cyan
            text = "RECOVERY CONFIRMED - RESUMING CONTROL"
            sound_active = False
            self.buzzer.stop()

        else:
            color = (108, 117, 125)
            text = "UNKNOWN"
            self.buzzer.stop()

        self.is_buzzer_active = sound_active
        return color, text, sound_active
