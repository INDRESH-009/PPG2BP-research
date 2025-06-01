//NESTED OBJECTS

//creating using object literal method all in one go - (most used,preffered)
const order = {
    id:1234,
    customers:{
        name:"Indresh",
        "age":20,
        address:{
            city:"Chennai",
            pin:600026
        }
    }
};

//creating using object literal method step by step
const order1 = {}
order1.id = 1234;
order1.customers = {}
order1.customers.name = "Indresh";
order1.customers.age = 20;
order1.customers.address = {}
order1.customers.address.city = "Chennai";
order1.customers.address.pin = 600026;


//creating using constructor method
const order2 = new Object();
order2.id = 1234;
order2.customers = new Object();
order2.customers.name = "Indresh";  
order2.customers.age = 20;
order2.customers.address = new Object();
order2.customers.address.city = "Chennai";
order2.customers.address.pin = 600026;






//READING A NESTED OBJECT

//using plain chaining - 
const pincode = order.customers.address.pin;
console.log(pincode); 

//using optional chaining 
const pincode2 = order.customers?.address?.pin;
console.log(pincode2);

//using destructuring 
const {age} = order?.customers; 
console.log(age); 


var {city="NA",pin} = order.customers?.address;
console.log(city,pin)


/* 
1. plain chaining    - simple , works as long as all the links are present .
                     - in the links if anyone of the link is absent or not initialised , it will return undefined.
                     - if we try to access the next link, it will throw an Typeerror as cannot read property of undefined.
                     - So typeError comes when we try to access property of undefined. if our chain stops exactly at undefined, it will not throw an error.
                     - eg : -order.customers.phone - will result in undefined.
                            -order.customers.phone.number - TypeError: Cannot read properties of undefined (reading 'number')
                            

2. optional chaining - ES6 feature 
                     - Optional chaining (?.) is just ordinary JavaScript code that keeps running even when something in the middle of a path is null or undefined.
                     - Safe. As soon as a link is null / undefined, the whole expression short‑circuits to undefined , so unlike plain chaining, it will not throw an error.
                     - It is a good practice to use optional chaining when we are not sure if the object is initialised or not(data from api is not gaurenteed , in such cases we can use it to prevent errors/crash).
                            - eg: order.customers?.phone?.number 
                            -will not throw an error, it will return undefined , as it shortcircuits to undefined at phone.

3. destructuring     - ES6 feature
                     - Destructuring is a JavaScript expression that makes it possible to unpack values from arrays, or properties from objects, into distinct variables.
                     - It is a good practice to use destructuring when we are sure that the object is initialised and we want to access multiple properties of the object.
                     - It is also used to avoid writing long code for accessing multiple properties of an object.
                     - It is also used to assign default values to the variables while destructuring.
                     
*/