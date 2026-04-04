# © 2024 Alec Fessler
# MIT License
# See LICENSE file in the project root for full license information.

import numpy as np
from PIL import Image

def main():
  SQUARE_SIZE_MM = 25
  CHESS_ROWS = 7
  CHESS_COLS = 10

  DPI = 300
  MM_PER_INCH = 25.4

  SQUARE_SIZE_PX = int(SQUARE_SIZE_MM * DPI / MM_PER_INCH)

  pattern_width = CHESS_COLS * SQUARE_SIZE_PX
  pattern_height = CHESS_ROWS * SQUARE_SIZE_PX

  pattern = np.zeros((pattern_height, pattern_width), dtype=np.uint8)

  for i in range(CHESS_ROWS):
    for j in range(CHESS_COLS):
      if (i + j) % 2 != 0: continue
      y1, y2 = i * SQUARE_SIZE_PX, (i + 1) * SQUARE_SIZE_PX
      x1, x2 = j * SQUARE_SIZE_PX, (j + 1) * SQUARE_SIZE_PX
      pattern[y1:y2, x1:x2] = 255

  border_size = int(SQUARE_SIZE_PX / 2)
  bordered_pattern = np.full(
    (pattern_height + 2 * border_size, pattern_width + 2 * border_size),
    255, dtype=np.uint8
  )
  bordered_pattern[border_size:border_size + pattern_height,
                   border_size:border_size + pattern_width] = pattern

  img = Image.fromarray(bordered_pattern, mode='L')
  img.save('chessboard_pattern.png', dpi=(DPI, DPI))
  print(f"Saved chessboard_pattern.png ({img.width}x{img.height} px, {DPI} DPI)")
  print(f"Print size: {img.width / DPI * MM_PER_INCH:.0f} x {img.height / DPI * MM_PER_INCH:.0f} mm")

if __name__ == "__main__":
  main()
