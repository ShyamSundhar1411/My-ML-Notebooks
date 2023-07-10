#include <Servo.h>
String my_data;
String my_intensity;
#include <Servo.h>
int brightness;
Servo myservo; 
Servo end_effector;
String receivedString = "";
String part1 = "";
String part2 = "";
void splitString(String receivedString, String& firstPart, String& secondPart) {
  int stringLength = receivedString.length();
  char charArray[stringLength + 1];
  receivedString.toCharArray(charArray, stringLength + 1);
  
  char* token = strtok(charArray, ";");
  
  if (token != NULL) {
    firstPart = String(token);
    token = strtok(NULL, ";");
    
    if (token != NULL) {
      secondPart = String(token);
    }
  }
}

void setup() {
  Serial.begin(115200);
  end_effector.attach(9);

}

void loop() {
  String my_data = Serial.readStringUntil('\r');
   int end_effector_angle = my_data.toInt();
   Serial.println("Part 1: " + part1);
   Serial.println("Part 2: " + part2);
  end_effector.write(45);
//  myservo.write(link1_angle);
}
