import pygame
import time

pygame.init()
pygame.mixer.init()

pygame.mixer.music.load("lagu/Take Me Back to London (feat. Stormzy) - Ed Sheeran.mp3")
pygame.mixer.music.play()

print("Lagu diputar...")
time.sleep(10)  # putar 10 detik
