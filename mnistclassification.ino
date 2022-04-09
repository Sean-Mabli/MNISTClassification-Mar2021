#include <MCUFRIEND_kbv.h>
MCUFRIEND_kbv tft;

#define YP A2
#define XM A3
#define YM 8
#define XP 9

#include <TouchScreen.h>
TouchScreen ts = TouchScreen(XP, YP, XM, YM, 300);

#define BLACK 0x0000
#define WHITE 0xFFFF

#define TS_MINX 130
#define TS_MAXX 905

#define TS_MINY 75
#define TS_MAXY 930

#define STATUS_X 10
#define STATUS_Y 65

bool drawing[54][40];

void setup(void) {
  Serial.begin(9600);

  tft.reset();

  uint16_t identifier = tft.readID();

  tft.begin(identifier);
  tft.setRotation(0);
  tft.fillScreen(BLACK);
  background();
}

#define MINPRESSURE 10
#define MAXPRESSURE 1000

void loop(void) {
  TSPoint p = ts.getPoint();
  if (p.z > MINPRESSURE && p.z < MAXPRESSURE)
  {
    p.x = map(p.x, TS_MAXX, TS_MINX, 480, 0);
    p.y = map(p.y, TS_MAXY, TS_MINY, 0, 320);

    pinMode(XM, OUTPUT);
    pinMode(YP, OUTPUT);

    if (p.x < 50 && 220 < p.y) {
      cleardrawing();
    }

    if (50 < p.x) {
      drawing[int(floor(p.y / 8) - 6)][int(floor(p.x / 8))] = true;
      tft.fillRect(floor(p.y / 8) * 8, floor(p.x / 8) * 8, 8, 8, WHITE);
    }
  }
}

void background() {
  tft.fillRect(0, 0, 320, 48, BLACK);
  tft.drawLine(0, 48, 320, 48, WHITE);
  tft.drawLine(220, 0, 220, 48, WHITE);

  tft.setTextColor(WHITE);
  tft.setTextSize(2);

  tft.setCursor(10, 18);
  tft.println("Prediction: ");

  tft.setCursor(240, 18);
  tft.println("Clear");
}

void cleardrawing() {
  tft.fillRect(0, 49, 320, 480, BLACK);
  delay(300);
  
  bool drawing[60][40] = {false};
}
