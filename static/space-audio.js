(() => {
  const audio = document.getElementById('space-audio');

  if (!(audio instanceof HTMLAudioElement)) {
    return;
  }

  audio.volume = 0.42;
  audio.loop = true;
  audio.autoplay = true;

  async function tryPlay() {
    try {
      await audio.play();
    } catch {
      // The browser may block autoplay until the first user gesture.
    }
  }

  function resumePlayback() {
    if (audio.paused) {
      tryPlay();
    }
  }

  document.addEventListener('DOMContentLoaded', tryPlay, { once: true });
  window.addEventListener('load', tryPlay, { once: true });
  document.addEventListener('pointerdown', resumePlayback, { passive: true });
  document.addEventListener('keydown', resumePlayback);
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) {
      resumePlayback();
    }
  });
})();
