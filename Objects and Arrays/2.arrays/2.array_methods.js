//ARRAY METHODS 

//1. Recognising an array - Array.isArray()
const arr = [10,20,30,40,50];
const greet = "hello";
let x = Array.isArray(arr);
let y = Array.isArray(greet);
console.log(x);
console.log(y);

//2. Length of an array - arr.length
console.log(arr.length);


//3. Adding an element to the end of an array - arr.push()
const arr2 = [10,20,30];
arr2.push(40);
arr2.push("hello");

let a = 32;
arr2.push(a);
console.log(arr2);


//4. Adding an element to the beginning of an array - arr.unshift()
const arr3 = ['a','b','c','d'];
arr3.unshift('z');
console.log(arr3);


//5. Removing an element from the end of an array - arr.pop()
const arr4 = [2,4,6,8,10];
arr4.pop();
console.log(arr4);


//6. Removing an element from the beginning of an array - arr.shift()
const arr5 = [1,2,3,4,5];
arr5.shift();
console.log(arr5);


//7. Insert , delete or replace anywhere in the array - arr.splice()





/* 
1. When you pull data from APIs, DOM calls, or third‑party libraries, use Array.isArray to avoid “length is undefined” errors.


*/