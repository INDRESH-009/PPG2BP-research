//OBJECT DESTRUCTURING
const user = {namee:"Alice",age:25,city:"New York"};

//normal destrcturing
const {namee,age} = user;
console.log(namee);
console.log(age);

// renaming the variables while destructuring
const {namee:fullName} = user;
console.log(fullName);

//default values while destructuring
const {role="guest"} = user;
console.log(role);

//nested Destructuring
const order = {
    id:1234,
    customer:{
        names:"Indresh",
        agee:20,
        address:{
            city:"Chennai",
            pin:600026
        }
    }
};
const {customer:{names,agee,address:{city,pin}}} = order;
console.log(names);
console.log(agee);
console.log(city);
console.log(pin);


//Using rest with objects 
const {customer:{names:userName,...restProps}} = order;
console.log(userName);
console.log(restProps);
