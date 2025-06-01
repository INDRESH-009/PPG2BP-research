//FUNCTION PARAMETER DESTRUCTURING 
//especially useful in react,API calls and utility function 

//Destructuring arrays which are functional parameters 
function sum([a,b]){
    return a+b;
}
console.log(sum([10,12]));

//Destructuring objects which are functional parameters 
function greet({name,age}){
    return `Hello ${name}, age ${age}`;
}
const obj = {
    name:"Indresh",
    gender:"male",
    age:20,
    state:"TN"
}
console.log(greet(obj));