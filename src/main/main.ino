
#define MOTOR_PIN_B_DIR 2
#define MOTOR_PIN_B_STEP 3

#define MOTOR_PIN_F_DIR 4
#define MOTOR_PIN_F_STEP 5

#define MOTOR_PIN_L_DIR 6
#define MOTOR_PIN_L_STEP 7

#define MOTOR_PIN_R_DIR 8
#define MOTOR_PIN_R_STEP 9

#define MOTOR_PIN_D_DIR 10
#define MOTOR_PIN_D_STEP 11

#define CMD_LENGTH 3

#define STEP_360 200
#define STEP_180 100
#define STEP_90 50

#define SPEED_FAST 1000
#define SPEED_SLOW 3000
#define WAIT_TIME 1000

#define uint8 unsigned char

char inputBuffer[CMD_LENGTH + 1];  // +1 for null terminator
uint8 index = 0;

void setup() {
  
  for(uint8 i = 0; i < 12 ; i++) {
      pinMode(i,OUTPUT);
  }

  Serial.begin(9600);
}

static inline void move_motor_face(uint8 dir_pin, uint8 step_pin, uint8 steps, bool up , short int speed){
  digitalWrite(dir_pin, up);
  
  for (int i = 0; i < steps; i++) {
    digitalWrite(step_pin, HIGH);
    delayMicroseconds(speed);
    digitalWrite(step_pin, LOW);
    delayMicroseconds(speed);
  }

  delayMicroseconds(WAIT_TIME);
}

/*
Replacement of U:
    - U = L R B2 F2 L' R'   D  L R B2 F2 L' R'
    - U' = L R B2 F2 L' R'   D'  L R B2 F2 L' R'
    - U2 = L R B2 F2 L' R'  D2  L R B2 F2 L' R'
*/
static inline void move_motor_U_face(bool right){
    move_motor_face(MOTOR_PIN_L_DIR , MOTOR_PIN_L_STEP, STEP_90, right ,SPEED_FAST);
    move_motor_face(MOTOR_PIN_R_DIR , MOTOR_PIN_R_STEP, STEP_90, right ,SPEED_FAST);
    move_motor_face(MOTOR_PIN_B_DIR , MOTOR_PIN_B_STEP, STEP_180, right ,SPEED_FAST);
    move_motor_face(MOTOR_PIN_F_DIR , MOTOR_PIN_F_STEP, STEP_180, right ,SPEED_FAST);
    move_motor_face(MOTOR_PIN_L_DIR , MOTOR_PIN_L_STEP, STEP_90, !right ,SPEED_FAST);
    move_motor_face(MOTOR_PIN_R_DIR , MOTOR_PIN_R_STEP, STEP_90, !right ,SPEED_FAST);
}

static inline void move_motor_U(){
    bool right = inputBuffer[2] - '0';
    move_motor_U_face(right);
    move_motor_face(MOTOR_PIN_D_DIR , MOTOR_PIN_D_STEP, STEP_90 * (inputBuffer[1]-'0')+ STEP_90, right , SPEED_FAST);
    move_motor_U_face(right);
}

static inline void move_motor(){
    char face = inputBuffer[0];
    bool step = inputBuffer[1] - '0';    // '1' → 1 or '0' -> 0
    bool dir = inputBuffer[2] - '0';     // '0' or '1'
    uint8 steps = step * STEP_90 + STEP_90;

     switch (face) {
        case 'D': move_motor_face(MOTOR_PIN_D_DIR , MOTOR_PIN_D_STEP , steps , dir , SPEED_SLOW); break;
        case 'F': move_motor_face(MOTOR_PIN_F_DIR , MOTOR_PIN_F_STEP , steps , dir , SPEED_SLOW); break;
        case 'B': move_motor_face(MOTOR_PIN_B_DIR , MOTOR_PIN_B_STEP , steps , dir , SPEED_SLOW); break;
        case 'L': move_motor_face(MOTOR_PIN_L_DIR , MOTOR_PIN_L_STEP , steps , dir , SPEED_SLOW); break;
        case 'R': move_motor_face(MOTOR_PIN_R_DIR , MOTOR_PIN_R_STEP , steps , dir , SPEED_SLOW); break;
        case 'U': move_motor_U(); break;
        default:
          Serial.println("Unknown face.");
          break;
    }
}

void loop() {

  if (index > 0) {
    move_motor();
    index = 0;
  }

  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == '\n') {
      if (index == CMD_LENGTH) {
        inputBuffer[CMD_LENGTH] = '\0'; // Null-terminate string
      } else {
        Serial.println("Invalid command length");
        index = 0; // Reset
      }
    } else {
      if (index < CMD_LENGTH) {
        inputBuffer[index++] = c;
      } else {
        // Buffer overflow, ignore and reset
        index = 0;
        Serial.println("Command too long");
      }
    }
  }
}

