// fast_controller_debug.ino
// Improved parser + debug for the Rubik motors controller.
//
// Protocol (same):
//  - "T\n"                => run test sequence (rotates each motor) and replies "OK\n"
//  - "<sequence>\n"       => tokens space separated: R, R', R2, U, U', U2, ...
//  - "DBG\n"              => toggles verbose debug prints
//
// Send commands from Serial Monitor (115200) and watch logs.

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

constexpr uint16_t STEP_90 = 800u; // pulses for 90°
enum GradeMove : uint8_t { G1 = 1u, G2 = 2u, G3 = 3u, G4 = 4u };

// microsecond delay for half-step (smaller => faster)
constexpr uint16_t STEP_DELAY_US = 65u;
// small settling time after finishing pulses (ms)
constexpr uint16_t SETTLE_MS = 100u;
// delay between enabling and disabling (ms)
constexpr uint16_t WAIT_TIME_MS = 20u;

constexpr uint16_t LINE_BUF = 1024;
char lineBuffer[LINE_BUF];
uint16_t lineIndex = 0;

struct Motor {
  uint8_t dirPin;
  uint8_t stepPin;
  uint8_t enablePin;

  void begin() const {
    pinMode(dirPin, OUTPUT);
    pinMode(stepPin, OUTPUT);
    pinMode(enablePin, OUTPUT);
    digitalWrite(enablePin, HIGH); // disable driver (active LOW)
    digitalWrite(stepPin, LOW);
    digitalWrite(dirPin, LOW);
  }

  void enable(bool on) const {
    digitalWrite(enablePin, on ? LOW : HIGH);
  }

  // Move with optional holdEnable to avoid toggling enable for macros
  void move(bool dir, enum GradeMove grade, bool holdEnable = false) const {
    digitalWrite(dirPin, dir ? HIGH : LOW);
    if (!holdEnable) digitalWrite(enablePin, LOW); // enable active-low

    uint16_t pulses = grade * STEP_90;    
    for (uint16_t i = 0u; i < pulses; ++i) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(STEP_DELAY_US);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(STEP_DELAY_US);
    }

    delay(SETTLE_MS);
    if (!holdEnable) digitalWrite(enablePin, HIGH); // disable
    if (!holdEnable) delay(WAIT_TIME_MS);
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

void enable_all() {
  faceToMotor('L')->enable(true);
  faceToMotor('R')->enable(true);
  faceToMotor('F')->enable(true);
  faceToMotor('D')->enable(true);
  faceToMotor('B')->enable(true);
}
void disable_all() {
  faceToMotor('L')->enable(false);
  faceToMotor('R')->enable(false);
  faceToMotor('F')->enable(false);
  faceToMotor('D')->enable(false);
  faceToMotor('B')->enable(false);
}

/* U replacement macro: perform the equivalent sequence keeping enable pins held.
   dir: true = positive, false = negative
   step_mult: 1 => 90°, 2 => 180°.
*/
void move_motor_u(bool dir, enum GradeMove grade) {
  Motor* l = faceToMotor('L');
  Motor* r = faceToMotor('R');
  Motor* f = faceToMotor('F');
  Motor* d = faceToMotor('D');
  Motor* b = faceToMotor('B');

  //enable_all();
  //delay(5);
  // U: L R B2 F2 L' R' 
  l->move(dir, G1);
  Serial.println("OK");

  r->move(dir, G1);
  Serial.println("OK");

  b->move(dir, G2);
  Serial.println("OK");

  f->move(dir, G2);
  Serial.println("OK");

  l->move(!dir, G1);
  Serial.println("OK");

  r->move(!dir, G1);
  Serial.println("OK");

  d->move(dir, grade);
  Serial.println("OK");

  l->move(dir, G1);
  Serial.println("OK");

  r->move(dir, G1);
  Serial.println("OK");

  b->move(dir, G2);
  Serial.println("OK");

  f->move(dir, G2);
  Serial.println("OK");

  l->move(!dir, G1);
  Serial.println("OK");

  r->move(!dir, G1);
  Serial.println("OK");

  // Pequeño settle y luego desactivar drivers
  //delay(SETTLE_MS);
  //disable_all();
  //delay(WAIT_TIME_MS);
}

void process_line(const char* line) {
  uint16_t len = strlen(line);
  if (len == 0) return;

  if (line[0] == 'T') {
    Motor* motors[5] = { &M_D, &M_F, &M_B, &M_L, &M_R };
    for (auto& m : motors) {
      m->move(true, G4);
      m->move(false, G4);
    }
    Serial.println("OK");
    return;
  }

  uint16_t i = 0;
  while (i < len) {
    while (i < len && line[i] == ' ') i++;
    if (i >= len) break;

    char face = line[i++];
    char mod = '\0';
    if (i < len && (line[i] == '\'' || line[i] == '2')) {
      mod = line[i++];
    }

    if (face == 'U') {
      bool dir = (mod != '\''); // ' => inverse
      GradeMove g = (mod == '2') ? G2 : G1;
      move_motor_u(dir, g);
      continue;
    }

    Motor* m = faceToMotor(face);
    bool dir = (mod != '\'');
    GradeMove g = (mod == '2') ? G2 : G1;
    m->move(dir, g);
  }

  Serial.println("OK");
}

void setup() {
  Serial.begin(115200);
  M_F.begin(); M_D.begin(); M_R.begin(); M_B.begin(); M_L.begin();
  delay(50);
}

void manual_test(){ 
  // Example test: execute sample sequence
  const char *test = "R R R R L2 L2 D' D' D' D' F2 F' F' B2 B2 U2 U' U U'\n";
  strncpy(lineBuffer, test, LINE_BUF-1);
  lineBuffer[LINE_BUF-1] = '\0';
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
      lineBuffer[lineIndex++] = c;
    }
  }
}
