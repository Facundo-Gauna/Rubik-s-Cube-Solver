// Arduino Mega sketch - improved protocol with calibration & sequence parsing
#include <Arduino.h>
#include <EEPROM.h>

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

// Default calibration (steps per 90°) — initial values; will be overwritten by EEPROM if present
constexpr uint16_t DEFAULT_STEPS_PER_90 = 800; // from original STEP_90 , 3200 = 360 grade 

constexpr uint16_t STEP_DELAY_FAST = 40;    // microseconds per half-step delay (smaller = faster)
constexpr uint16_t WAIT_TIME_MS = 550;

constexpr uint16_t LINE_BUF = 512;
char lineBuffer[LINE_BUF];
uint16_t lineIndex = 0;

struct Motor {
  uint8_t dirPin, stepPin, enablePin;
  uint16_t steps_per_90; // configurable
  
  void begin() const {
    pinMode(dirPin, OUTPUT);
    pinMode(stepPin, OUTPUT);
    pinMode(enablePin, OUTPUT);
    digitalWrite(enablePin, HIGH); // disabled
  }

  void move(bool dir, uint16_t steps) const {
    digitalWrite(dirPin, dir ? HIGH : LOW);
    digitalWrite(enablePin, LOW); // enable motor (active LOW)
    for (uint16_t i = 0; i < steps; ++i) {
      digitalWrite(stepPin, HIGH);
      delayMicroseconds(STEP_DELAY_FAST);
      digitalWrite(stepPin, LOW);
      delayMicroseconds(STEP_DELAY_FAST);
    }
    delay(1);
    digitalWrite(enablePin, HIGH); // disable
    delay(WAIT_TIME_MS);
  }
};

Motor M_F = {MOTOR_PIN_F_DIR, MOTOR_PIN_F_STEP, MOTOR_PIN_F_ENABLE, DEFAULT_STEPS_PER_90};
Motor M_D = {MOTOR_PIN_D_DIR, MOTOR_PIN_D_STEP, MOTOR_PIN_D_ENABLE, DEFAULT_STEPS_PER_90};
Motor M_R = {MOTOR_PIN_R_DIR, MOTOR_PIN_R_STEP, MOTOR_PIN_R_ENABLE, DEFAULT_STEPS_PER_90};
Motor M_B = {MOTOR_PIN_B_DIR, MOTOR_PIN_B_STEP, MOTOR_PIN_B_ENABLE, DEFAULT_STEPS_PER_90};
Motor M_L = {MOTOR_PIN_L_DIR, MOTOR_PIN_L_STEP, MOTOR_PIN_L_ENABLE, DEFAULT_STEPS_PER_90};

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

constexpr uint8_t NUM_FACES = 5;
char facesOrder[NUM_FACES] = {'D','F','B','L','R'};

void save_calibration_to_eeprom() {
  int addr = 0; // EEPROM ADDRESS
  for (int i = 0; i < NUM_FACES; ++i) {
    Motor* m = faceToMotor(facesOrder[i]);
    uint16_t v = m->steps_per_90;
    EEPROM.put(addr, v);
    addr += sizeof(uint16_t);
  }
}

void load_calibration_from_eeprom() {
  int addr = 0; // EEPROM ADDRESS
  for (int i = 0; i < NUM_FACES; ++i) {
    uint16_t v;
    EEPROM.get(addr, v);
    if (v == 0xFFFF || v == 0) { // not set or erased
      v = DEFAULT_STEPS_PER_90;
    }
    Motor* m = faceToMotor(facesOrder[i]);
    if (m) m->steps_per_90 = v;
    addr += sizeof(uint16_t);
  }
}

void sendOK() { Serial.println("OK"); }
void sendErr(const char* msg) { Serial.print("ERR "); Serial.println(msg); }


bool parse_and_execute_token(const String& token) {
  if (token.length() == 0) return true;

  char face = token.charAt(0);
  bool mult = token.charAt(1);    // 0 = 90 deg, 1 = 180 deg.
  bool dir = token.charAt(2)-'0'; // choose encoding: original used inputBuffer[2] - '0' with 0/1; keep dir true==positive

  Motor* m = faceToMotor(face);
  m->move(dir, m->steps_per_90*steps+m->steps_per_90);

  return true;
}

// Parse a sequence string of tokens separated by spaces and execute in order
bool process_sequence_tokens(const String& seq) {
  // Tokenize by spaces
  int start = 0;
  int len = seq.length();
  while (start < len) {
    while (start < len && isspace(seq.charAt(start))) start++;
    if (start >= len) break;
    int end = start;
    while (end < len && !isspace(seq.charAt(end))) end++;
    String token = seq.substring(start, end);
    // Ensure uppercase and trim
    token.toUpperCase();
    if (!parse_and_execute_token(token)) {
      return false;
    }
    start = end;
  }
  return true;
}

/* Replacement of U: 
- U = L R B2 F2 L' R' D L R B2 F2 L' R' 
- U' = L R B2 F2 L' R' D' L R B2 F2 L' R' 
- U2 = L R B2 F2 L' R' D2 L R B2 F2 L' R' 
*/ 
void move_motor_u(bool dir, uint8_t step_mult){ 
  Motor* l = faceToMotor('L');
  Motor* r = faceToMotor('R');
  Motor* f = faceToMotor('F');
  Motor* d = faceToMotor('D');
  Motor* b = faceToMotor('B');

  l->move(dir, l->steps_per_90);
  r->move(dir, r->steps_per_90);
  b->move(dir, b->steps_per_90<<1);
  f->move(dir, f->steps_per_90<<1);
  l->move(!dir, l->steps_per_90);
  r->move(!dir, r->steps_per_90);

  d->move(dir, step_mult*d->steps_per_90 + d->steps_per_90);
  
  l->move(dir, l->steps_per_90);
  r->move(dir, r->steps_per_90);
  b->move(dir, b->steps_per_90<<1);
  f->move(dir, f->steps_per_90<<1);
  l->move(!dir, l->steps_per_90);
  r->move(!dir, r->steps_per_90);
}

void process_line(const char* line) {

  switch(line[0]){
    case 'T':{ // Test all motors 360 steps
      uint8_t idx = 0;
      while(idx<5){
        Motor* m = faceToMotor(facesOrder[idx++]);
        m->move(true,m->steps_per_90<<2);
        m->move(false,m->steps_per_90<<2);
      }
      return;
    }
    case 'G':{ // Get calibration from line[1]: motor
      Motor* m = faceToMotor(facesOrder[line[1]]);
      Serial.print(line[1]);
      Serial.print(": ");
      Serial.println(m->steps_per_90);
      return;
    }
    case 'S':{ // Set steps calibrated for motor. <S><Motor_Face><Steps> 
      Motor* m = faceToMotor(facesOrder[line[1]]);
      uint16_t steps = line[2]-'0', len = line.length, idx = 3;
      while(idx < len && isdigit(line[idx]) ){
        steps = (steps*10) + (line[idx++]-'0');
      }
      m->steps_per_90 = steps;
      save_calibration_to_eeprom();
      return
    }
    default:{ // its a sequence or a single move. example : F' D2 R' D' F' U' F U F2 R' F2 U2 R2 F2 R2 F2 U2 F2 D' F2
      uint16_t s = 0, len = line.length;
      while(s < len){
        
        while(s < len && line[s] == ' ') s++;
        if(s==len) break;

        char face = line[s++];
        //
        //
        //
        //
        //
        //
        //
        //

        if(s == len || line[s] == ' '){
          s++; continue;
        } 
        char dirOrsteps = line[s++];
        if(face == 'U') move_motor_u()

        s++;
      }
      break;
    }
  }

  // Split first token
  int sp = s.indexOf(' ');
  String cmd = (sp == -1) ? s : s.substring(0, sp);
  cmd.toUpperCase();

  if (cmd == "SEQ") {
    // rest of line is moves
    String rest = (sp == -1) ? String("") : s.substring(sp + 1);
    if (rest.length() == 0) { sendErr("No moves"); return; }
    Serial.println("BUSY");
    bool ok = process_sequence_tokens(rest);
    if (ok) sendOK(); else sendErr("Sequence failed");
    return;
  }

  if (cmd == "MOVE") {
    // Expected: MOVE <face> <90|180|270> <dir>
    if (sp == -1) { sendErr("MOVE args"); return; }
    String args = s.substring(sp + 1);
    // parse tokens
    char face = 0;
    int angle = 90;
    int dir = 1;
    int p = 0;
    // token1 face
    while (p < (int)args.length() && isspace(args.charAt(p))) p++;
    if (p >= (int)args.length()) { sendErr("MOVE args"); return; }
    face = args.charAt(p);
    p++;
    // read angle
    while (p < (int)args.length() && isspace(args.charAt(p))) p++;
    int startAngle = p;
    while (p < (int)args.length() && !isspace(args.charAt(p))) p++;
    if (p > startAngle) {
      angle = args.substring(startAngle, p).toInt();
    }
    // read dir
    while (p < (int)args.length() && isspace(args.charAt(p))) p++;
    if (p < (int)args.length()) {
      dir = args.substring(p).toInt();
    }
    int mult = 1;
    if (angle == 90) mult = 1;
    else if (angle == 180) mult = 2;
    else if (angle == 270) mult = 3;
    else { sendErr("Bad angle"); return; }
    Serial.println("BUSY");
    execute_face_token(face, mult, dir != 0);
    sendOK();
    return;
  }

}

void setup() {
  Serial.begin(115200);

  M_F.begin(); M_D.begin(); M_R.begin(); M_B.begin(); M_L.begin();

  load_calibration_from_eeprom();

  delay(50);
}

void loop() {
  while (Serial.available() > 0) {
    char c = (char)Serial.read();
    if (c == '\n') {
      lineBuffer[lineIndex] = '\0';
      process_line(lineBuffer);
      lineIndex = 0;
    } else {
      if (lineIndex < LINE_BUF - 1) {
        lineBuffer[lineIndex++] = c;
      } else {
        lineIndex = 0;
        Serial.println("ERR input overflow");
      }
    }
  }
}
