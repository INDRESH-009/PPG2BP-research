//ARRAY DESTRUCTURING 
/* 
It is a syntax that allows you to unpack values from arrays or extract properties from objects into distinct variables
*/

const arr = [10,20,30];

//destrcturing in order
const [a,b] = arr;
console.log(a);
console.log(b);

//destrcturing by skipping values
const[first, ,third] = arr;
console.log(first);
console.log(third);

//destrcturing in reverse order
const [x,y,z] = arr.reverse();
console.log(x);
console.log(y); 
console.log(z);

//destrcturing by having default values 
// if a value is missing or undefined , the default value is taken 
const[p=5,q=10,c=15,d=100] = arr;
console.log(p);
console.log(q);
console.log(c);
console.log(d);

//swapping variables using destructuring
let m=1,n=2;
[m,n]=[n,m];
console.log(m,n);

//Using rest operator with arrrays while destructuring
const[head,...tail] = arr;
console.log(head);
console.log(tail);

