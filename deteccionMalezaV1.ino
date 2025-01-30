#define speedPinR 9    //  RIGHT PWM pin connect ENA
#define RightMotorDirPin1  12    //Right Motor direction pin 1 to IN1 
#define RightMotorDirPin2  11    //Right Motor direction pin 2 IN2
#define speedPinL 6    // Left PWM pin connect MODEL-X ENB
#define LeftMotorDirPin1  7    //Left Motor direction pin 1 to IN3 
#define LeftMotorDirPin2  8   //Left Motor direction pin 1 to  IN4 

/*motor control*/
void go_MotoGuadana(void)  //MotoGuadana ON
{
  digitalWrite(RightMotorDirPin1, HIGH);
  digitalWrite(RightMotorDirPin2,LOW);
  digitalWrite(LeftMotorDirPin1,HIGH);
  digitalWrite(LeftMotorDirPin2,LOW);
  analogWrite(speedPinL,200);
  analogWrite(speedPinR,200);
}

void stop_Stop()    //Stop MotoGuadana
{
  digitalWrite(RightMotorDirPin1, LOW);
  digitalWrite(RightMotorDirPin2,LOW);
  digitalWrite(LeftMotorDirPin1,LOW);
  digitalWrite(LeftMotorDirPin2,LOW);
}

/*set motor speed */
void set_Motorspeed(int speed_L,int speed_R)
{
  analogWrite(speedPinL,speed_L); 
  analogWrite(speedPinR,speed_R);    
}
void init_GPIO()
{
	pinMode(RightMotorDirPin1, OUTPUT); 
	pinMode(RightMotorDirPin2, OUTPUT); 
	pinMode(speedPinL, OUTPUT);  
 
	pinMode(LeftMotorDirPin1, OUTPUT);
  pinMode(LeftMotorDirPin2, OUTPUT); 
  pinMode(speedPinR, OUTPUT); 
	stop_Stop();
}
void setup() {
    Serial.begin(9600); // Configura la velocidad del puerto serial
    init_GPIO();// Inicializa los puertos de para motores
}

void loop() {
    if (Serial.available() > 0) { // Si hay datos en el buffer serial
        char comando = Serial.read(); // Lee el carácter enviado
        if (comando == 'H') {
           
            go_MotoGuadana();//Activa la motoGuadana
        } else if (comando == 'L') {
            
            stop_Stop();//Stop
        }
    }
}