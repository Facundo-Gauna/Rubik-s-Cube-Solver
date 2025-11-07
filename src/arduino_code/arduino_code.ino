// fast_controller.ino
// Arduino Mega sketch — simplified protocol (no calibration), sends OK after operations.
// Protocol:
//  - "T\n"                => run test sequence (rotates each motor) and replies "OK\n"
//  - "<sequence>\n"       => sequence of tokens (space separated), replies "OK\n" when finished
// Tokens: R, R', R2, U, U', U2, etc.

#include <Arduino.h>

constexpr uint8_t MOTOR_PIN_F_DIR = 55;
constexpr uint8_t MOTOR_PIN_F_STEP = 54;
constexpr uint8_t MOTOR_PIN_F_ENABLE = 38;

constexpr uint8_t MOTOR_PIN_D_DIR = 61;
constexpr uint8_t MOTOR_PIN_D_STEP = 60;
constexpr uint8_t MOTOR_PIN_D_ENABLE = 56;

constexpr uint8_t MOTOR_PIN_R_DIR = 48;
constexpr uint8_t MOTOR_PIN_R_STEP = 46;
constexpr uint8_t MOTOR_PIN_R_ENABLE = 62;

constexpr uint8_t MOTOR_PIN_B_DIR = 34;
constexpr uint8_t MOTOR_PIN_B_STEP = 36;
constexpr uint8_t MOTOR_PIN_B_ENABLE = 30;

constexpr uint8_t MOTOR_PIN_L_DIR = 28;
constexpr uint8_t MOTOR_PIN_L_STEP = 26;
constexpr uint8_t MOTOR_PIN_L_ENABLE = 24;

constexpr uint16_t STEP_90 = 800u;
enum GradeMove : uint8_t { G1 = 1u, G2 = 2u, G3 = 3u, G4 = 4u };

// microsecond delay for half-step (smaller => faster)
constexpr uint16_t STEP_DELAY_US = 65u;
// small settling time after finishing pulses (ms)
constexpr uint16_t SETTLE_MS = 100u;
// delay between enabling and disabling (ms)
constexpr uint16_t WAIT_TIME_MS = 100u;

constexpr size_t LINE_BUF = 512;
char lineBuffer[LINE_BUF];
size_t lineIndex = 0;

struct Motor {
  uint8_t dirPin;
  uint8_t stepPin;
  uint8_t enablePin;

  void begin() const {
    pinMode(dirPin, OUTPUT);
    pinMode(stepPin, OUTPUT);
    pinMode(enablePin, OUTPUT);
    digitalWrite(enablePin, HIGH); // disable driver (A4988/DRV active LOW)
  }

  void move(bool dir, uint32_t steps) const {
    digitalWrite(dirPin, dir);
    digitalWrite(enablePin, LOW); // enable (active LOW)

    for (uint32_t i = 0u; i < steps; ++i) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(STEP_DELAY_US);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(STEP_DELAY_US);
    }

    delay(SETTLE_MS);
    digitalWrite(enablePin, HIGH); // disable
    delay(WAIT_TIME_MS);
  }
};

Motor M_F = {MOTOR_PIN_F_DIR, MOTOR_PIN_F_STEP, MOTOR_PIN_F_ENABLE};
Motor M_D = {MOTOR_PIN_D_DIR, MOTOR_PIN_D_STEP, MOTOR_PIN_D_ENABLE};
Motor M_R = {MOTOR_PIN_R_DIR, MOTOR_PIN_R_STEP, MOTOR_PIN_R_ENABLE};
Motor M_B = {MOTOR_PIN_B_DIR, MOTOR_PIN_B_STEP, MOTOR_PIN_B_ENABLE};
Motor M_L = {MOTOR_PIN_L_DIR, MOTOR_PIN_L_STEP, MOTOR_PIN_L_ENABLE};

Motor* faceToMotor(char face) {
  switch (face) {
    case 'D': return &M_D;
    case 'F': return &M_F;
    case 'B': return &M_B;
    case 'L': return &M_L;
    case 'R': return &M_R;
    default: return nullptr;
  }
}

/* U replacement:
   U  -> L R B2 F2 L' R'  D  L R B2 F2 L' R'
   U' -> same but D'
   U2 -> same but D2
   The function expects dir (true=positive, false=negative) and
   step_mult: 0 -> 90°, 1 -> 180° (keeps your original mapping)
*/
void move_motor_u(bool dir, uint8_t step_mult) {
  Motor* l = faceToMotor('L');
  Motor* r = faceToMotor('R');
  Motor* f = faceToMotor('F');
  Motor* d = faceToMotor('D');
  Motor* b = faceToMotor('B');

  l->move(dir, (uint32_t)G1 * STEP_90);
  r->move(dir, (uint32_t)G1 * STEP_90);
  b->move(dir, (uint32_t)G2 * STEP_90);
  f->move(dir, (uint32_t)G2 * STEP_90);
  l->move(!dir, (uint32_t)G1 * STEP_90);
  r->move(!dir, (uint32_t)G1 * STEP_90);

  d->move(dir, (uint32_t)step_mult * STEP_90);

  l->move(dir, (uint32_t)G1 * STEP_90);
  r->move(dir, (uint32_t)G1 * STEP_90);
  b->move(dir, (uint32_t)G2 * STEP_90);
  f->move(dir, (uint32_t)G2 * STEP_90);
  l->move(!dir, (uint32_t)G1 * STEP_90);
  r->move(!dir, (uint32_t)G1 * STEP_90);
}

void process_line(const char* line) {
  size_t len = lineIndex;
  if (len == 0) return;
  
  Serial.print(len);
  Serial.print(": ");
  Serial.println(line);

  // Test command -> T\n
  if (line[0] == 'T') {
    /* Motors positions : 
      Back(white) Left(red) Down(green) Right(orange) Front(yellow) 
    */
    Motor* motors[5] = { &M_D, &M_F, &M_B, &M_L, &M_R };
    for (uint8_t i = 0; i < 5; ++i) {
      motors[i]->move(true, (uint32_t)G4 * STEP_90);   // forward
      motors[i]->move(false, (uint32_t)G4 * STEP_90);  // back
    }
    Serial.println("OK");
    return;
  }

  size_t s = 0;
  while (s < len) {
    while (s < len && line[s] == ' ') ++s;
    if (s >= len) break;

    char face = line[s++];

    if (s >= len || line[s] == ' ') {
      Motor* m = faceToMotor(face);
      if( face == 'U') move_motor_u(true,1);
      else m->move(true, (uint32_t)G1 * STEP_90);
      continue;
    }

    char mod = line[s++];
    if (face == 'U') {
      move_motor_u( (mod == '\'') , ((mod == '2') ? 2 : 1));
    } else {
      Motor* m = faceToMotor(face);
      uint16_t stepss = (mod == '2') ? G2 : G1;
      m->move((mod != '\''), STEP_90 * stepss);
    }
  }

  Serial.println("OK");
}

void setup() {
  Serial.begin(115200);
  M_F.begin(); M_D.begin(); M_R.begin(); M_B.begin(); M_L.begin();
  delay(50);
}

void manual_test(){ 
  lineBuffer[0] = 'U';
  lineIndex = 1;
  process_line(lineBuffer);
}

void loop() {
  //manual_test();
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      lineBuffer[lineIndex] = '\0';
      process_line(lineBuffer);
      lineIndex = 0;
    } else {
      if (lineIndex < (LINE_BUF - 1)) {
        lineBuffer[lineIndex++] = c;
      } else {
        // overflow
        lineIndex = 0;
        Serial.println("ERR input overflow");
      }
    }
  }
}
