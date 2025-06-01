//ARRAY OF OBJECTS 

// Its a datastrcture where each element of an array is an object 
const product = [
    {id:1,name:"Book",price:100,"hello greet":"hi"},
    {id:2,name:"Pen",price:20},
    {id:3,name:"Notebook",price:60}
];

//ACCESSING PROPERTIES

//1. access individual objects using array indices and then access the properties using dot notation
console.log(product[0].name);
console.log(product[1].price);

//2. access individual objects using array indices and then access the properties using bracket notation(when the key is dynamic or not an valid identifier)
console.log(product[0]["hello greet"]);

let key = "price";
console.log(product[0][key]);


//ITERATING OVER THE ARRAY OF OBJECTS 

//1. using for loop 
for(let i=0;i<product.length;i++){
    console.log(product[i].name);
}

//using for...of loop
for(let pro of product){
    console.log(`${pro.name} costs $${pro.price}`);
}

//using forEach()
product.forEach(pro=>{
    console.log(pro.name.toUpperCase());
    console.log(pro.price);
});

//TRANSFORMING DATA - using map()

// its a built in function in js viz used to transform or manipulate the elements of an array. 
// It creates a new array by applying a callback function to  each element of the original array 

const names = product.map(p=>p.name);
console.log(names); 
//['Book', 'Pen', 'Notebook']

// returing a new transformed object using map()
const withGST = product.map(p=>({
    ...p,
    priceWithGST: (p.price *1.18).toFixed(2)
}));
console.log(withGST);


//FILTERING OBJECTS - using filter()
// it creates a new array with all elements that pass the test implemented by the provided function.
const expensive = product.filter(p=>p.price>50);
console.log(expensive);

//FIND MATCHING OBJECT - using find()
const pen = product.find(p=>p.name === "Pen");
console.log(pen);

//VALIDATE DATA - using every() and some()

//every() - checks if all elements in the array pass the test implemented by the provided function
const allHavePrice = product.every(p=>p.price>0);
console.log(allHavePrice);

//some() - checks if at least one element in the array passes the test implemented by the provided function
const hasExpensive = product.some(p=>p.price>100);
console.log(hasExpensive);


//AGGREGATE DATA - using reduce() :
// It is a powerful array method which aggregates or reduces an array into a single value. 
// It takes a callback fucntion and an iniial value as arguments. 
// The callback function takes two arguments: an accumulator and the current value.
// (sum,p)=>sum+p.price --> is the callback function where sum is the accumulator and p is the current item from product array. 
// 0 is the inital value of the sum(accumulator) and then it increases

const total = product.reduce((sum,p)=>sum+p.price,0);
console.log(`Total Price : $${total}`);


//MODIFYING A SPECIFIC OBJECT - using map() and findIndex()

//using findIndex() - change price of pen to 25
const index = product.findIndex(p=>p.id===2);
if(index!== -1){
    product[index].price = 25;
}
console.log(product[1]);

//using map() 
const updatedProduct = product.map(p=>
    p.id===2 ? {...p,price:25}:p
);
console.log(updatedProduct);

//SORTING ARRAY OF OBJECTS - using sort()
/* 
It sorts the array in place and returns the sorted array.
It takes a compare function as an argument which defines the sort order.
 (a,b)=>a.price-b.price ---- is the compare function
 a = {id:1,name:"Book",price:100} and b = {id:2,name:"Pen",price:20}
a.price-b.price = 100-20 = 80
The sort function interprects results like this :

| Return Value from Function | Meaning                            |
| -------------------------- | ---------------------------------- |
| < 0                        | a comes before b                   |
| > 0                        | a comes after b                    |
|   0                        | a and b stay in the same order     |

*/ 

//ascending order of price
product.sort((a,b)=>a.price-b.price);
console.log(product);

//desending order of price 
product.sort((a,b)=>b.price-a.price);
console.log(product);

//ordering by name 
product.sort((a,b)=> a.name.localeCompare(b.name));
console.log(product);

//DESTRUCTURING INSIDE ITERATION
for(const{name,price} of product){
    console.log(`${name}:$${price}`);
}


//REAL WORLD USE CASE 
//suppose you are buiding a product cataloug ina ecom app and want to show the most appordable product 
const items = [
    {name:"Samsung A10",price:10000},
    {name:"iphone 14",price:80000},
    {name:"Micromax",price:5000},
]
const listAffordableProducts = (productList,budget)=>{
    return productList
        .filter(p=>p.price<=budget)
        .map(p=>p.name);
}
const affordableProducts = listAffordableProducts(items,20000);
console.log(affordableProducts);
























