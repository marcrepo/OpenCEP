grammar Port;

program: event+ ;
event: 'event' ID FH ;
ID : [a-zA-Z][a-zA-Z_0-9]+ ;
FH : [0-9]+ ;      
WS : [ \t\r\n]+ -> skip ;