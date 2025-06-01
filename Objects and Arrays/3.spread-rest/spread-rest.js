//SPREAD WITH ARRAYS 

//cloning an array
const newarr = [...arr,6,7,8,9,10,11,12,13,14,15];
console.log(newarr);

//function arguments using spread operator
function sum(a,b,c){
    return a+b+c;
}
const values = [10,20,30];
const result = sum(...values);
console.log(result);

//merging 2 arrays using spread operator
const arr1 = [1,2,3];
const arr2 = [4,5,6];
const merged = [...arr1,...arr2];
console.log(merged);



//SPREAD WITH OBJECTS

//cloning an object
const user = {name:"indresh",age:10,role:"billionaire"};
const company = {...user}
console.log(user);
console.log(company);

//updating an object
const updated = {...company,role:"ceo",employee:1000000};
console.log(updated);

//merging 2 objects
const location = {city:"Amsterdam",country:"netherlands"};
const final = {...updated,...location};
console.log(final);

//if keys overlap the last one will be taken 
const user1 = {name:"indresh",age:10,role:"billionire"};
const updated2 = {...user1,role:"ceo"};
console.log(updated2);





//REST OPERATOR 
//collects multiple elements and condenses them into a single element
//it is used in function arguments

//rest in function parameters
function multiply(multiplier,...numbers){
    return numbers.map(n=>n*multiplier);
}
let num = [1,2,3,4,5];
let xy = multiply(2,...num);

//here 3 is the multipler and 1,2,3,4,5 are the numbers . the ...num gathers the remaining arguments into an array
let xyz = multiply(3,1,2,3,4,5);
console.log(xy);
console.log(xyz);

//array destrcturing using rest operator
const value = [10,20,30,40];
const [first,...rest] = value;
console.log(first); //10
console.log(rest);  //[20,30,40]

//object destructuring using rest operator
const user3 = {
    namee: "Bob",
    age:20,
    email:"bob@email.com"
};
const {namee,...contactInfo} = user3;
console.log(namee);
console.log(contactInfo);


//spread operator : expands an iterable (like an array) into more elements
//rest operator : condenses multiple elements into a single element

//spread operator is used in array,objects and fucntions
//rest operator is used in function parameters and destructuring

//spread operator is used to copy or merge
//rest operator is used to collect remaining elements in an array or object