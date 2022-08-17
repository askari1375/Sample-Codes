#include <Servo.h>

Servo myservo1;  // create servo object to control a servo
Servo myservo2; 
// twelve servo objects can be created on most boards

int pos = 0;    // variable to store the servo position
int posi=0;
int del = 0;

void setup() {
  myservo1.attach(9);  // attaches the servo on pin 9 to the servo object
  myservo2.attach(10);  // attaches the servo on pin 9 to the servo object
}

void loop() {


for(del=5;del<30; del+=5){  // Varies the delay between servo commands, i.e. speed of motion
        for (posi = 10; posi <= 50; posi += 10) { // goes from 0 degrees to 180 degrees
        myservo2.write(posi);              // tell servo to go to position in variable 'pos'
        delay(del);                       // waits 15ms for the servo to reach the position
     
      
      for (pos = 50; pos <= 120; pos += 1) { // goes from 0 degrees to 180 degrees
        // in steps of 1 degree
        myservo1.write(pos);              // tell servo to go to position in variable 'pos'
        delay(del);                       // waits 15ms for the servo to reach the position
      }
      for (pos = 120; pos >= 50; pos -= 1) { // goes from 180 degrees to 0 degrees
        myservo1.write(pos);              // tell servo to go to position in variable 'pos'
        delay(del);                       // waits 15ms for the servo to reach the position
      }

    }
  }
}
