//CREATING OBJECTS IN JS 

//object literal method (fast and most used)
const user = {
    name:"Indresh",
    age:20,
    "fav language":"JS"
};

//constructor method (rarely used)
const user2 = new Object();
user2.name = "Indresh";
user2.age = 20,
user2["fav language"] = "JS";

/* 
literal method - key-value pair is used to create an object.
               - key is a string and , incase u dont give it a string, it will be converted to string. - name->"name" , age->"age"
               - fav language -> "fav language" (space is not allowed in key, so we have to use "" to create a key with space)



constructor method - new Object() is used to create an object.
                   - once the object is created, we can add key-value pairs to it using dot notation or bracket notation.
                   - bracket notation is used when the key has space or special characters.
                   - dot notation is used when the key is a valid identifier (no space, no special characters).

*/









//ACCESSING AND UPDATING OBJECT PROPERTIES

//dot notation
console.log(user.name); 
console.log(user.age);

//bracket notation 
console.log(user["name"]);
console.log(user["fav language"]);

const k = "age";
console.log(user[k]);

//updating the object properties - access the key and reassign
user.name = "Indresh Kumar";
user.age += 1;
user["fav language"] = "JavaScript";



/* 
dot notation - access keys which are static 
             - access keys which are valid identifiers (no space, no special characters).

bracket notation - access dynamic keys 
                 - also used when the key is stored in a variable.
                 - also used when the key is not a valid identifier (has space or special characters).
               
*/


