grammar Port;

program: event+ ;
event: 'event' ID ;
ID : [a-zA-Z][a-zA-Z_0-9]+ ;      
WS : [ \t\r\n]+ -> skip ;