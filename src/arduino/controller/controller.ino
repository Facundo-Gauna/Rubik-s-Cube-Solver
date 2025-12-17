// controller.ino
// Parser for the Rubik motors controller.
//
// Protocol (same):
//  - "T\n"                => run test sequence (rotates each motor) and replies "OK\n"
//  - "S<ms time>"         => change the micro seconds time to complete a move
//  - "<sequence>\n"       => tokens space separated: R, R', R2, U, U', U2, ...
//
// Send commands from Serial Monitor (115200) and watch logs.

#include <Arduino.h>

// The drivers are setted in a ramps 1.4 which uses these pins.
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

constexpr uint16_t STEP_90 = 800u; // pulses for 90° (usage of microstepping)
enum GradeMove : uint8_t { G1 = 1u, G2 = 2u, G3 = 3u, G4 = 4u };

uint16_t step_delay_us = 65u;          // microsec delay for half-step (smaller => faster)
uint16_t settle_ms = 100u;             // small settling time after finishing pulses (ms)
uint16_t wait_time_ms = 20u;           // delay between enabling and disabling (ms)

constexpr uint16_t MIN_STEP_DELAY_US = 8u;
constexpr uint16_t MAX_STEP_DELAY_US = 2000u;

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
      delayMicroseconds(step_delay_us);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(step_delay_us);
    }

    delay(settle_ms);
    if (!holdEnable) digitalWrite(enablePin, HIGH); // disable
    if (!holdEnable) delay(wait_time_ms);
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
  l->move(dir, G1);

  r->move(dir, G1);

  b->move(dir, G2);

  f->move(dir, G2);

  l->move(!dir, G1);

  r->move(!dir, G1);

  d->move(dir, grade);

  l->move(dir, G1);

  r->move(dir, G1);

  b->move(dir, G2);

  f->move(dir, G2);

  l->move(!dir, G1);

  r->move(!dir, G1);

  //delay(SETTLE_MS);
  //disable_all();
  //delay(WAIT_TIME_MS);
}

void set_move_duration_ms(uint16_t duration_ms) {
  // duration_ms = desired time for a G1 (90°) move, in ms.
  // Compute step_delay_us so that a G1 move (800 pulses) lasts ~ duration_ms.
  uint32_t overhead_ms = (uint32_t)settle_ms + (uint32_t)wait_time_ms;

  if (duration_ms <= overhead_ms + 1) {
    step_delay_us = MIN_STEP_DELAY_US;
  } else {
    uint32_t pulses_time_us = (uint32_t)(duration_ms - overhead_ms) * 1000u;
    uint32_t sd = pulses_time_us / (  1 << STEP_90);
    if (sd < MIN_STEP_DELAY_US) sd = MIN_STEP_DELAY_US;
    if (sd > MAX_STEP_DELAY_US) sd = MAX_STEP_DELAY_US;
    step_delay_us = (uint16_t)sd;
  }

  Serial.print("SOK "); Serial.print(duration_ms); Serial.print("ms -> step_delay_us=");
  Serial.println(step_delay_us);
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

  if (line[0] == 'S') {
    uint32_t ms_time = 0;
    uint16_t i = 1;
    while (i < len && line[i] >= '0' && line[i] <= '9') {
      ms_time = ms_time * 10 + (uint8_t)(line[i] - '0');
      ++i;
    }
    if (ms_time > 0 && ms_time <= 60000) {
      set_move_duration_ms((uint16_t)ms_time);

      Serial.print("STEP_DELAY_US=");
      Serial.println(step_delay_us);
    } else {
      Serial.println("SERR"); // invalid
    }
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
